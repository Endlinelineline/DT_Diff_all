"""kW-space layered evaluation and reproducible Guangning result figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator

from utils.wind_metrics import LayeredWindMetrics, to_sample_turbine_horizon


FIGURE_NAMES = (
    "turbine_history_forecast_4x4.png",
    "per_turbine_acc30.png",
    "per_turbine_errors.png",
    "per_horizon_overall_macro.png",
    "site_power_timeseries.png",
    "turbine_horizon_absolute_error_heatmap.png",
    "prediction_truth_scatter.png",
    "per_turbine_error_boxplot.png",
)


def _set_time_axis(axis) -> None:
    locator = AutoDateLocator(minticks=3, maxticks=6)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _save_figure(fig, path: Path, subtitle: str) -> None:
    fig.text(0.5, 0.012, subtitle, ha="center", va="bottom", fontsize=9)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _checkpoint_epoch(checkpoint: Path | None) -> int | None:
    if checkpoint is None or not checkpoint.exists():
        return None
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = state.get("epoch")
    return int(epoch) + 1 if epoch is not None else None


def _validate_truth_alignment(truth_nth, dataset, anchors: np.ndarray) -> None:
    for position in sorted({0, len(anchors) // 2, len(anchors) - 1}):
        anchor = int(anchors[position])
        expected = dataset.values_kw[anchor + 1:anchor + 1 + truth_nth.shape[2]].T
        if not np.allclose(truth_nth[position], expected, rtol=0, atol=0.1):
            raise ValueError(
                f"Truth at sample {position} does not match the raw-kW dataset. "
                "Check array layout, split, stride, and inverse scaling."
            )


def _metric_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _plot_value(value) -> float:
    return np.nan if value is None else float(value)


def _metric_label(value, suffix: str) -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def _select_example_index(dataset, anchors, horizon: int, history_minutes: int):
    """Choose an active-power example without looking at prediction errors."""
    values = np.asarray(dataset.values_kw)
    starts = np.maximum(anchors - history_minutes + 1, 0)
    ends = anchors + horizon + 1
    middle = (len(anchors) - 1) / 2
    sums = np.vstack((np.zeros((1, values.shape[1])), values.cumsum(axis=0)))
    square_sums = np.vstack((
        np.zeros((1, values.shape[1])), np.square(values).cumsum(axis=0)
    ))
    for minimum_kw in (100.0, 0.0):
        invalid_minute = np.any(~np.isfinite(values) | (values <= minimum_kw), axis=1)
        invalid_prefix = np.concatenate(([0], np.cumsum(invalid_minute)))
        eligible = np.flatnonzero(invalid_prefix[ends] == invalid_prefix[starts])
        if len(eligible):
            lengths = (ends[eligible] - starts[eligible])[:, None]
            means = (sums[ends[eligible]] - sums[starts[eligible]]) / lengths
            variances = (
                (square_sums[ends[eligible]] - square_sums[starts[eligible]])
                / lengths - np.square(means)
            )
            minimum_std = np.sqrt(np.maximum(variances, 0)).min(axis=1)
            varying = minimum_std > 20.0
            variation_rule = 20.0 if varying.any() else None
            if varying.any():
                eligible = eligible[varying]
            chosen = int(eligible[np.argmin(np.abs(eligible - middle))])
            return chosen, minimum_kw, variation_rule, int(len(eligible))
    raise ValueError(
        "No zero-free 16-turbine history/future window is available for the "
        "example figure; choose a different data range instead of plotting "
        "a shutdown-like interval as a normal example."
    )


def create_wind_report(
    prediction,
    truth,
    *,
    source_layout: str,
    dataset,
    output_dir: Path,
    figure_dir: Path,
    anchor_indices=None,
    model_name: str = "DiffDiff",
    checkpoint: Path | None = None,
    seed: int | None = None,
    evaluation_name: str = "standard_test",
    example_history_minutes: int = 180,
) -> dict:
    """Create the complete report without modifying model predictions.

    `prediction` and `truth` must already be inverse-scaled to kW. We verify
    three truth windows against the original kW cache before calculating.
    """
    prediction_nth = to_sample_turbine_horizon(prediction, source_layout)
    truth_nth = to_sample_turbine_horizon(truth, source_layout)
    if prediction_nth.shape != truth_nth.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {prediction_nth.shape} vs {truth_nth.shape}")
    if prediction_nth.shape[0] < 1:
        raise ValueError("No forecast samples supplied")
    n, turbine_count, horizon = prediction_nth.shape
    names = list(dataset.feature_names)
    if turbine_count != len(names) or horizon != dataset.pred_len:
        raise ValueError(
            f"Expected [N,{len(names)},{dataset.pred_len}], got {prediction_nth.shape}"
        )
    if anchor_indices is None:
        if n > len(dataset.sample_starts):
            raise ValueError("More predictions than dataset windows")
        anchors = dataset.border_start + dataset.sample_starts[:n] + dataset.seq_len - 1
    else:
        anchors = np.asarray(anchor_indices, dtype=np.int64)
        if anchors.shape != (n,):
            raise ValueError("anchor_indices must contain one history-end index per sample")
    if np.any(anchors < dataset.seq_len - 1) or np.any(anchors + horizon >= len(dataset.timestamps)):
        raise ValueError("Forecast anchors exceed the available raw data")
    _validate_truth_alignment(truth_nth, dataset, anchors)
    (
        example_index, example_minimum_kw,
        example_variation_kw, eligible_example_count,
    ) = _select_example_index(dataset, anchors, horizon, example_history_minutes)
    anchor = int(anchors[example_index])
    history_start = max(anchor - example_history_minutes + 1, 0)

    output_dir = Path(output_dir).resolve()
    figure_dir = Path(figure_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    deliverables = [
        output_dir / name for name in (
            "metrics.json", "per_turbine_metrics.csv", "per_horizon_metrics.csv",
            "site_metrics.csv", "predictions.npz", "curve_example_data.csv",
            "site_endpoint_series.csv", "turbine_horizon_absolute_error.csv",
            "scatter_plot_data.csv", "error_boxplot_data.csv", "figure_data_manifest.json",
        )
    ] + [figure_dir / name for name in FIGURE_NAMES]
    existing = [str(path) for path in deliverables if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing report files: {existing[:3]}")

    metric_engine = LayeredWindMetrics(names, horizon)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        metric_engine.update(prediction_nth[start:stop], truth_nth[start:stop])
    metrics = metric_engine.result()
    epoch = _checkpoint_epoch(checkpoint)
    metadata = {
        "dataset": "Guangning 16-turbine 1-minute power",
        "model": model_name,
        "prediction_steps": horizon,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_epoch": epoch,
        "random_seed": seed,
        "evaluation_name": evaluation_name,
        "array_layout": "[num_samples, num_turbines, prediction_steps]",
        "source_array_layout": source_layout,
        "power_unit": "kW",
        "sample_count": n,
        "curve_example": {
            "sample_index": example_index,
            "history_start": str(dataset.timestamps[history_start]),
            "history_end": str(dataset.timestamps[anchor]),
            "future_end": str(dataset.timestamps[anchor + horizon]),
            "selection": "nearest to test midpoint among windows whose 16 actual turbine powers stay above the threshold throughout displayed history and future; require per-turbine variation when available",
            "minimum_actual_power_kW_exclusive": example_minimum_kw,
            "minimum_each_turbine_std_kW_exclusive": example_variation_kw,
            "eligible_window_count": eligible_example_count,
        },
    }
    metrics = {"metadata": metadata, **metrics}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)

    turbine_rows = []
    for scope in ("curve", "endpoint"):
        turbine_rows.extend(
            {"scope": scope, **row} for row in metrics[f"per_turbine_{scope}"]
        )
    _metric_frame(turbine_rows).to_csv(output_dir / "per_turbine_metrics.csv", index=False)
    horizon_rows = []
    for item in metrics["per_horizon"]:
        for scope in ("curve_overall", "macro_turbine_curve", "site_overall"):
            horizon_rows.append({
                "prediction_step": item["prediction_step"], "scope": scope,
                **item[scope],
            })
    _metric_frame(horizon_rows).to_csv(output_dir / "per_horizon_metrics.csv", index=False)
    _metric_frame([
        {"scope": scope, **metrics["site_overall"][scope]}
        for scope in ("curve", "endpoint")
    ]).to_csv(output_dir / "site_metrics.csv", index=False)

    timestamps = dataset.timestamps.values
    future_times = timestamps[anchors[:, None] + np.arange(1, horizon + 1)[None, :]]
    release_times = timestamps[anchors]
    with TemporaryDirectory(prefix="wind_report_", dir=output_dir) as temporary:
        temporary_npz = Path(temporary) / "predictions.npz"
        with open(temporary_npz, "wb") as handle:
            np.savez(
                handle, prediction_kw=prediction_nth, truth_kw=truth_nth,
                turbine_ids=np.asarray(names), target_timestamps=future_times,
                history_end_timestamps=release_times,
            )
        os.replace(temporary_npz, output_dir / "predictions.npz")

    step_ids = np.arange(1, horizon + 1)
    heatmap_values = np.divide(
        metric_engine.abs_sum, metric_engine.finite,
        out=np.full_like(metric_engine.abs_sum, np.nan),
        where=metric_engine.finite > 0,
    )
    heatmap_rows = [
        {"turbine_id": name, "prediction_step": int(step),
         "mean_absolute_error_kW": float(heatmap_values[turbine, step - 1]),
         "finite_points": int(metric_engine.finite[turbine, step - 1])}
        for turbine, name in enumerate(names) for step in step_ids
    ]
    _metric_frame(heatmap_rows).to_csv(
        output_dir / "turbine_horizon_absolute_error.csv", index=False
    )

    site_truth = np.asarray(truth_nth[:, :, -1], dtype=np.float64).sum(axis=1)
    site_prediction = np.asarray(prediction_nth[:, :, -1], dtype=np.float64).sum(axis=1)
    site_frame = pd.DataFrame({
        "target_timestamp": future_times[:, -1],
        "truth_site_kW": site_truth,
        "prediction_site_kW": site_prediction,
    })
    site_frame.to_csv(output_dir / "site_endpoint_series.csv", index=False)

    history_times = timestamps[history_start:anchor + 1]
    history_values = dataset.values_kw[history_start:anchor + 1]
    curve_rows = []
    for turbine, name in enumerate(names):
        for time, power in zip(history_times, history_values[:, turbine]):
            curve_rows.append({"sample_index": example_index,
                               "turbine_id": name, "timestamp": time,
                               "series": "history", "power_kW": power})
        for step in range(horizon):
            curve_rows.append({"sample_index": example_index,
                               "turbine_id": name, "timestamp": future_times[example_index, step],
                               "series": "truth_future", "power_kW": truth_nth[example_index, turbine, step]})
            curve_rows.append({"sample_index": example_index,
                               "turbine_id": name, "timestamp": future_times[example_index, step],
                               "series": "prediction_future", "power_kW": prediction_nth[example_index, turbine, step]})
    _metric_frame(curve_rows).to_csv(output_dir / "curve_example_data.csv", index=False)

    # The two dense diagnostic charts use deterministic plotted subsets; their
    # corresponding exact plotted values are saved separately from full arrays.
    rng = np.random.default_rng(2021)
    scatter_size = min(60_000, n * turbine_count * horizon)
    flat_indices = rng.choice(n * turbine_count * horizon, scatter_size, replace=False)
    sample_ids, remainder = divmod(flat_indices, turbine_count * horizon)
    turbine_ids, horizon_ids = divmod(remainder, horizon)
    scatter_frame = pd.DataFrame({
        "sample_index": sample_ids,
        "turbine_id": np.asarray(names)[turbine_ids],
        "prediction_step": horizon_ids + 1,
        "truth_kW": truth_nth[sample_ids, turbine_ids, horizon_ids],
        "prediction_kW": prediction_nth[sample_ids, turbine_ids, horizon_ids],
    })
    scatter_frame.to_csv(output_dir / "scatter_plot_data.csv", index=False)
    box_rows = []
    box_sample_ids = rng.choice(n, min(n, 2_000), replace=False)
    for turbine, name in enumerate(names):
        for sample in box_sample_ids:
            box_rows.append({
                "sample_index": int(sample), "turbine_id": name,
                "absolute_error_kW": float(abs(
                    prediction_nth[sample, turbine, -1] - truth_nth[sample, turbine, -1]
                )),
            })
    box_frame = _metric_frame(box_rows)
    box_frame.to_csv(output_dir / "error_boxplot_data.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12,
                         "axes.labelsize": 10, "figure.titlesize": 14})
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, turbine_count))
    subtitle = (
        f"Dataset: Guangning | Model: {model_name} | H={horizon} min | "
        f"checkpoint epoch: {epoch if epoch is not None else 'unknown'} | "
        f"seed: {seed if seed is not None else 'unknown'}"
    )
    # 1. History and future curves. Both future curves start at the last actual
    # history value, so the plotted boundary is visually continuous.
    fig, axes = plt.subplots(4, 4, figsize=(20, 13), sharex=True)
    for turbine, axis in enumerate(axes.flat):
        anchor_value = history_values[-1, turbine]
        x_future = np.concatenate(([timestamps[anchor]], future_times[example_index]))
        actual = np.concatenate(([anchor_value], truth_nth[example_index, turbine]))
        predicted = np.concatenate(([anchor_value], prediction_nth[example_index, turbine]))
        axis.plot(history_times, history_values[:, turbine], color="#607D8B", lw=1.15, label="History")
        axis.plot(x_future, actual, color="#1565C0", lw=1.4, label="Truth future")
        axis.plot(x_future, predicted, color="#EF6C00", lw=1.4, label="Predicted future")
        axis.axvline(timestamps[anchor], color="gray", ls="--", lw=0.8)
        axis.set_title(names[turbine], color=colors[turbine])
        axis.set_ylabel("Power (kW)")
        axis.grid(alpha=0.2)
        _set_time_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("Timestamp")
    fig.suptitle(
        f"History and {horizon}-minute forecast, sample {example_index} "
        f"(all actual powers > {example_minimum_kw:g} kW)"
    )
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.972))
    fig.tight_layout(rect=(0, 0.026, 1, 0.94))
    _save_figure(fig, figure_dir / FIGURE_NAMES[0], subtitle)

    # 2. Turbine Acc30 curve and endpoint comparison.
    x = np.arange(turbine_count)
    curve = metrics["per_turbine_curve"]
    endpoint = metrics["per_turbine_endpoint"]
    fig, axis = plt.subplots(figsize=(16, 6))
    axis.bar(x - 0.19, [_plot_value(row["strict_acc30_pct"]) for row in curve], 0.38,
             color=colors, alpha=0.95, label="Full curve")
    axis.bar(x + 0.19, [_plot_value(row["strict_acc30_pct"]) for row in endpoint], 0.38,
             color=colors, alpha=0.45, label="Minute-60 endpoint")
    axis.set_xticks(x, names, rotation=45)
    axis.set_xlabel("Turbine ID")
    axis.set_ylabel("Strict Acc30 (%)")
    axis.set_title("Strict Acc30 by turbine")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_figure(fig, figure_dir / FIGURE_NAMES[1], subtitle)

    # 3. Error measures by turbine. Turbine colors are stable across panels.
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    for axis, key, label in zip(axes, ("MAE_kW", "RMSE_kW", "MAPE_pct"),
                                ("MAE (kW)", "RMSE (kW)", "MAPE (%)")):
        axis.bar(x - 0.19, [_plot_value(row[key]) for row in curve], 0.38,
                 color=colors, alpha=0.95, label="Full curve")
        axis.bar(x + 0.19, [_plot_value(row[key]) for row in endpoint], 0.38,
                 color=colors, alpha=0.45, label="Minute-60 endpoint")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_title("Error measures by turbine")
    axes[0].legend()
    axes[-1].set_xticks(x, names, rotation=45)
    axes[-1].set_xlabel("Turbine ID")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_figure(fig, figure_dir / FIGURE_NAMES[2], subtitle)

    # 4. Horizon variation of pooled and equal-turbine metrics.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, key, label in zip(axes.flat, ("strict_acc30_pct", "MAE_kW", "RMSE_kW", "MAPE_pct"),
                                ("Strict Acc30 (%)", "MAE (kW)", "RMSE (kW)", "MAPE (%)")):
        axis.plot(step_ids, [_plot_value(row["curve_overall"][key]) for row in metrics["per_horizon"]],
                  color="#C00017", lw=2, label="Overall (pooled)")
        axis.plot(step_ids, [_plot_value(row["macro_turbine_curve"][key]) for row in metrics["per_horizon"]],
                  color="#1565C0", lw=2, label="Macro turbine")
        axis.set_title(label)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    for axis in axes[-1]:
        axis.set_xlabel("Prediction step (minute)")
    fig.suptitle("Overall and macro metrics across the forecast horizon")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save_figure(fig, figure_dir / FIGURE_NAMES[3], subtitle)

    # 5. Full endpoint series for the site sum, with endpoint site metrics.
    site_metrics = metrics["site_overall"]["endpoint"]
    fig, axis = plt.subplots(figsize=(17, 6))
    axis.plot(site_frame["target_timestamp"], site_truth / 1_000,
              color="#1565C0", lw=0.8, label="True site power")
    axis.plot(site_frame["target_timestamp"], site_prediction / 1_000,
              color="#EF6C00", lw=0.8, alpha=0.82, label="Predicted site power")
    axis.set_title("Site total power at the final forecast step")
    axis.set_ylabel("Total power (MW)")
    axis.set_xlabel("Target timestamp")
    _set_time_axis(axis)
    axis.grid(alpha=0.2)
    axis.legend()
    axis.text(0.01, 0.03,
              f"Strict Acc30={_metric_label(site_metrics['strict_acc30_pct'], '%')}  "
              f"RMSE={_metric_label(site_metrics['RMSE_kW'], ' kW')}  "
              f"MAPE={_metric_label(site_metrics['MAPE_pct'], '%')}",
              transform=axis.transAxes,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_figure(fig, figure_dir / FIGURE_NAMES[4], subtitle)

    # 6. Full-curve mean absolute error, indexed by turbine and lead minute.
    fig, axis = plt.subplots(figsize=(15, 7))
    image = axis.imshow(heatmap_values, aspect="auto", cmap="magma", origin="upper")
    axis.set_yticks(np.arange(turbine_count), names)
    axis.set_xlabel("Prediction step (minute)")
    axis.set_ylabel("Turbine ID")
    axis.set_title("Mean absolute error by turbine and prediction step")
    axis.set_xticks(np.linspace(0, horizon - 1, min(7, horizon)).astype(int),
                    np.linspace(1, horizon, min(7, horizon)).astype(int))
    fig.colorbar(image, ax=axis, label="Mean absolute error (kW)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_figure(fig, figure_dir / FIGURE_NAMES[5], subtitle)

    # 7. Deterministically sampled points; y=x is a direct-power reference.
    fig, axis = plt.subplots(figsize=(10, 8))
    for turbine, name in enumerate(names):
        points = scatter_frame.loc[scatter_frame["turbine_id"] == name]
        axis.scatter(points["truth_kW"], points["prediction_kW"],
                     s=3, alpha=0.14, color=colors[turbine], label=name,
                     rasterized=True)
    finite_scatter = scatter_frame[["truth_kW", "prediction_kW"]].to_numpy()
    finite_scatter = finite_scatter[np.isfinite(finite_scatter).all(axis=1)]
    lower = min(0.0, float(finite_scatter.min()))
    upper = float(finite_scatter.max())
    axis.plot([lower, upper], [lower, upper], "k--", lw=1.5, label="y = x")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("True power (kW)")
    axis.set_ylabel("Predicted power (kW)")
    axis.set_title(f"True versus predicted power ({scatter_size:,} plotted points)")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 0.83, 1))
    _save_figure(fig, figure_dir / FIGURE_NAMES[6], subtitle)

    # 8. Endpoint error distribution, from the rows saved in the companion CSV.
    fig, axis = plt.subplots(figsize=(16, 6))
    groups = [box_frame.loc[box_frame["turbine_id"] == name, "absolute_error_kW"].to_numpy()
              for name in names]
    boxes = axis.boxplot(groups, tick_labels=names, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
    axis.set_title(f"Endpoint absolute error by turbine ({len(box_sample_ids):,} windows each)")
    axis.set_xlabel("Turbine ID")
    axis.set_ylabel("Absolute error (kW)")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_figure(fig, figure_dir / FIGURE_NAMES[7], subtitle)

    manifest = {
        "plots": {FIGURE_NAMES[0]: ["curve_example_data.csv"],
                  FIGURE_NAMES[1]: ["per_turbine_metrics.csv"],
                  FIGURE_NAMES[2]: ["per_turbine_metrics.csv"],
                  FIGURE_NAMES[3]: ["per_horizon_metrics.csv"],
                  FIGURE_NAMES[4]: ["site_endpoint_series.csv", "site_metrics.csv"],
                  FIGURE_NAMES[5]: ["turbine_horizon_absolute_error.csv"],
                  FIGURE_NAMES[6]: ["scatter_plot_data.csv"],
                  FIGURE_NAMES[7]: ["error_boxplot_data.csv"]},
        "prediction_values": "predictions.npz",
        "figure_directory": str(figure_dir),
        "scatter_and_boxplot_sampling_seed": 2021,
        "curve_example_selection": metadata["curve_example"],
    }
    with open(output_dir / "figure_data_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return metrics


def report_saved_predictions(args) -> dict:
    """Postprocess existing kW arrays without another diffusion inference."""
    from data_provider.data_factory import data_provider
    from exp.exp_long_term_forecasting import load_run_config, resolve_checkpoint

    checkpoint = resolve_checkpoint(args.checkpoint)
    config = load_run_config(checkpoint)["data_config"].copy()
    if args.data_root:
        config["root_path"] = args.data_root
    if args.cache_path:
        config["data_path"] = args.cache_path
    if args.test_stride:
        config["test_stride"] = args.test_stride
    config["rebuild_cache"] = False
    dataset, _ = data_provider(config, "test")
    prediction = np.load(args.predictions, mmap_mode="r")
    truth = np.load(args.truth, mmap_mode="r")
    if args.sample_start < 0 or args.sample_start >= len(prediction):
        raise ValueError("--sample_start must be within the saved prediction array")
    if args.sample_limit is not None:
        if args.sample_limit < 1:
            raise ValueError("--sample_limit must be positive")
    stop = min(
        len(prediction),
        args.sample_start + args.sample_limit if args.sample_limit is not None else len(prediction),
    )
    prediction = prediction[args.sample_start:stop]
    truth = truth[args.sample_start:stop]
    anchors = (
        dataset.border_start
        + dataset.sample_starts[args.sample_start:stop]
        + dataset.seq_len - 1
    )
    output_dir = Path(args.results_dir)
    figure_dir = Path(args.figures_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a fresh results directory: {output_dir}")
    if figure_dir.exists() and any(figure_dir.iterdir()):
        raise FileExistsError(f"Choose a fresh figure directory: {figure_dir}")
    result = create_wind_report(
        prediction, truth, source_layout=args.source_layout, dataset=dataset,
        output_dir=output_dir, figure_dir=figure_dir,
        anchor_indices=anchors, model_name=args.model,
        checkpoint=checkpoint, seed=args.seed,
        evaluation_name=(
            f"saved_predictions_samples_{args.sample_start}_{stop}"
            if args.sample_limit is not None or args.sample_start else "saved_predictions_full_test"
        ),
    )
    print(json.dumps({
        "metrics_file": str((output_dir / "metrics.json").resolve()),
        "figure_directory": str(figure_dir.resolve()),
        "shape": result["shape"],
        "curve_overall": result["curve_overall"],
    }, ensure_ascii=False, indent=2))
    return result
