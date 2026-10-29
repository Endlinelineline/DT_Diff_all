"""15-minute publication-point evaluation for Guangning DiffDiff.

Every forecast uses real observations ending at the publication timestamp.
Predictions are never fed back into the next input window.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from lightning.fabric import seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp.exp_long_term_forecasting import (
    Exp_Long_Term_Forecast,
    load_model_and_dataset,
)
from utils.wind_metrics import WindMetricAccumulator


def _inverse(values: torch.Tensor, dataset) -> np.ndarray:
    mean = torch.as_tensor(dataset.scaler.mean_, device=values.device, dtype=values.dtype)
    scale = torch.as_tensor(dataset.scaler.scale_, device=values.device, dtype=values.dtype)
    return (values * scale + mean).detach().cpu().numpy()


def _release_indices(dataset, seq_len: int, pred_len: int, interval: int):
    test_start, test_end = dataset.split_bounds["test"]
    first = max(test_start - 1, seq_len - 1)
    indices = [
        index
        for index in range(first, test_end)
        if dataset.timestamps[index].minute % interval == 0
        and dataset.timestamps[index].second == 0
        and index + 1 >= test_start
        and index + pred_len < test_end
        and dataset.timestamps[index + pred_len]
        - dataset.timestamps[index - seq_len + 1]
        == pd.Timedelta(minutes=seq_len + pred_len - 1)
    ]
    if not indices:
        raise ValueError("No valid 15-minute publication points fit in the test split")
    return indices


def _plot_endpoint_curve(
    frame: pd.DataFrame, metrics: dict, output_path: Path, model_name: str
):
    prediction_columns = [
        column for column in frame.columns if column.startswith("prediction_")
    ]
    truth_columns = [column for column in frame.columns if column.startswith("truth_")]
    prediction_mw = frame[prediction_columns].sum(axis=1) / 1000.0
    truth_mw = frame[truth_columns].sum(axis=1) / 1000.0
    figure, axis = plt.subplots(figsize=(15, 5))
    axis.plot(frame["target_time"], truth_mw, label="Truth", linewidth=1.0)
    axis.plot(frame["target_time"], prediction_mw, label=model_name, linewidth=1.0)
    axis.set_title("Guangning 60-minute endpoint forecast at 15-minute releases")
    axis.set_xlabel("Target time")
    axis.set_ylabel("Wind-farm power (MW)")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    acc30 = metrics["Acc30"]
    acc30_label = "n.a." if acc30 is None else f"{acc30:.2f}%"
    axis.text(
        0.01,
        0.02,
        f"Acc30={acc30_label}  MAE={metrics['MAE_kW']:.2f} kW",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_endpoint_diagnostics(
    frame: pd.DataFrame, output_dir: Path, model_name: str
) -> None:
    """Visualize comparable observed and forecast power at 60-minute endpoints."""
    names = [column.removeprefix("prediction_").removesuffix("_kw")
             for column in frame if column.startswith("prediction_")]
    predictions = frame[[f"prediction_{name}_kw" for name in names]].to_numpy()
    truth = frame[[f"truth_{name}_kw" for name in names]].to_numpy()
    errors = np.abs(predictions - truth)

    zoom_end = frame["target_time"].iloc[0] + pd.Timedelta(days=3)
    zoom = frame.loc[frame["target_time"] < zoom_end]
    zoom_metrics = WindMetricAccumulator()
    zoom_metrics.update(
        zoom[[f"prediction_{name}_kw" for name in names]].to_numpy(),
        zoom[[f"truth_{name}_kw" for name in names]].to_numpy(),
    )
    _plot_endpoint_curve(
        zoom,
        zoom_metrics.result(),
        output_dir / "rolling_endpoint_wind_farm_first_3d.png",
        model_name,
    )

    turbine_zoom = frame.loc[
        frame["target_time"] < frame["target_time"].iloc[0] + pd.Timedelta(days=2)
    ]
    figure, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    for channel, (axis, name) in enumerate(zip(axes.flat, names)):
        axis.plot(
            turbine_zoom["target_time"], turbine_zoom[f"truth_{name}_kw"],
            label="Observed", linewidth=1.1,
        )
        axis.plot(
            turbine_zoom["target_time"], turbine_zoom[f"prediction_{name}_kw"],
            label="Predicted", linewidth=1.0, alpha=0.85,
        )
        axis.set_title(name.replace("turbine_", "Turbine "))
        axis.set_ylabel("kW")
        axis.grid(alpha=0.25)
    figure.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2,
    )
    figure.suptitle("Observed vs predicted power: first 48 test hours", y=0.995)
    figure.autofmt_xdate()
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_dir / "rolling_endpoint_turbines_first_2d.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    maximum = float(max(np.nanmax(predictions), np.nanmax(truth), 1.0))
    minimum = float(min(np.nanmin(predictions), np.nanmin(truth), 0.0))
    density = axis.hexbin(truth.ravel(), predictions.ravel(), gridsize=75, mincnt=1)
    axis.plot([minimum, maximum], [minimum, maximum], "k--", linewidth=1, label="Perfect")
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed power (kW)")
    axis.set_ylabel("Predicted power (kW)")
    axis.set_title("All turbines: observed vs predicted at minute 60")
    axis.legend()
    figure.colorbar(density, ax=axis, label="Endpoint count")
    figure.tight_layout()
    figure.savefig(output_dir / "rolling_endpoint_scatter.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4))
    axis.bar([name.removeprefix("turbine_") for name in names], errors.mean(axis=0))
    axis.set_xlabel("Turbine")
    axis.set_ylabel("MAE (kW)")
    axis.set_title("Minute-60 endpoint MAE by turbine")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "rolling_endpoint_mae_by_turbine.png", dpi=160)
    plt.close(figure)

    hourly_mae = pd.Series(errors.mean(axis=1)).groupby(frame["target_time"].dt.hour).mean()
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(hourly_mae.index, hourly_mae.values, marker="o")
    axis.set_xticks(range(24))
    axis.set_xlabel("Target hour of day")
    axis.set_ylabel("MAE (kW)")
    axis.set_title("Minute-60 endpoint error by target hour")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "rolling_endpoint_mae_by_hour.png", dpi=160)
    plt.close(figure)


def _plot_horizon_examples(sample_frames: list[pd.DataFrame], output_path: Path) -> None:
    if not sample_frames:
        return
    figure, axes = plt.subplots(len(sample_frames), 1, figsize=(12, 3 * len(sample_frames)))
    axes = np.atleast_1d(axes)
    for axis, sample in zip(axes, sample_frames):
        prediction = sample.filter(regex=r"^prediction_.*_kw$").sum(axis=1) / 1000.0
        truth = sample.filter(regex=r"^truth_.*_kw$").sum(axis=1) / 1000.0
        axis.plot(sample["lead_minute"], truth, label="Observed", linewidth=1.5)
        axis.plot(sample["lead_minute"], prediction, label="Predicted", linewidth=1.5)
        axis.set_title(f"Release {sample['release_time'].iloc[0]} (all 16 turbines)")
        axis.set_ylabel("Wind-farm MW")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Forecast lead minute")
    axes[0].legend()
    figure.suptitle("Example 60-minute forecast trajectories", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_training_history(checkpoint: Path, output_dir: Path) -> None:
    metrics_path = checkpoint.parent / "lightning_logs" / "version_0" / "metrics.csv"
    if not metrics_path.exists():
        return
    history = pd.read_csv(metrics_path)
    train = history.dropna(subset=["loss_epoch"]).groupby("epoch")["loss_epoch"].last()
    validation = history.dropna(subset=["val_loss"]).groupby("epoch")["val_loss"].last()
    if train.empty or validation.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(train.index + 1, train.values, marker="o", label="Training loss")
    axis.plot(
        validation.index + 1, validation.values, marker="o", label="Validation loss"
    )
    best_epoch = int(validation.idxmin()) + 1
    axis.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    axis.set_title(f"Training and validation loss (best epoch {best_epoch})")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_validation_loss.png", dpi=160)
    plt.close(figure)


def run_rolling_test(args, model_class) -> dict:
    seed_everything(args.seed, workers=True)
    output_dir = Path(args.results_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new empty rolling output directory: {output_dir}")
    (
        checkpoint,
        _run_config,
        data_config,
        model,
        dataset,
        _loader,
        device,
    ) = load_model_and_dataset(args, args.checkpoint, model_class)

    seq_len = int(data_config["seq_len"])
    label_len = int(data_config["label_len"])
    pred_len = int(data_config["pred_len"])
    interval = int(args.release_interval)
    releases = _release_indices(dataset, seq_len, pred_len, interval)
    batch_size = int(args.batch_size or data_config["batch_size"])
    all_horizon_metrics = WindMetricAccumulator()
    endpoint_metrics = WindMetricAccumulator()
    endpoint_rows = []
    sample_positions = set(np.linspace(0, len(releases) - 1, 3, dtype=int).tolist())
    sample_frames = []
    prediction_curves = np.empty(
        (len(releases), pred_len, len(dataset.feature_names)), dtype=np.float64
    )
    truth_curves = np.empty_like(prediction_curves)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(releases), batch_size):
            batch_indices = releases[batch_start:batch_start + batch_size]
            conditions = np.stack(
                [
                    dataset.scaled_values[index - seq_len + 1:index + 1]
                    for index in batch_indices
                ]
            )
            condition = torch.from_numpy(conditions).float().to(device)
            placeholder = torch.zeros(
                len(batch_indices),
                label_len + pred_len,
                len(dataset.feature_names),
                device=device,
            )
            samples = model.predict_step({"x": placeholder, "c": condition}, 0)
            point_scaled = torch.median(
                samples[:, :, label_len:, :].cpu(), dim=0
            ).values.to(device)
            prediction_kw = _inverse(point_scaled, dataset)
            truth_kw = np.stack(
                [dataset.values_kw[index + 1:index + pred_len + 1] for index in batch_indices]
            )
            stop = batch_start + len(batch_indices)
            prediction_curves[batch_start:stop] = prediction_kw
            truth_curves[batch_start:stop] = truth_kw
            all_horizon_metrics.update(prediction_kw, truth_kw)
            endpoint_metrics.update(prediction_kw[:, -1, :], truth_kw[:, -1, :])

            for local, release_index in enumerate(batch_indices):
                release_position = batch_start + local
                row = {
                    "release_time": dataset.timestamps[release_index],
                    "target_time": dataset.timestamps[release_index + pred_len],
                }
                for channel, name in enumerate(dataset.feature_names):
                    row[f"prediction_{name}_kw"] = float(prediction_kw[local, -1, channel])
                    row[f"truth_{name}_kw"] = float(truth_kw[local, -1, channel])
                endpoint_rows.append(row)

                if release_position in sample_positions:
                    sample = pd.DataFrame(
                        {
                            "release_time": dataset.timestamps[release_index],
                            "target_time": dataset.timestamps[
                                release_index + 1:release_index + pred_len + 1
                            ],
                            "lead_minute": np.arange(1, pred_len + 1),
                        }
                    )
                    for channel, name in enumerate(dataset.feature_names):
                        sample[f"prediction_{name}_kw"] = prediction_kw[local, :, channel]
                        sample[f"truth_{name}_kw"] = truth_kw[local, :, channel]
                    sample_frames.append(sample)
            print(
                f"Publication points: {min(batch_start + batch_size, len(releases))}/"
                f"{len(releases)}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_frame = pd.DataFrame(endpoint_rows)
    endpoint_frame.to_csv(output_dir / "rolling_endpoint_predictions.csv", index=False)
    if sample_frames:
        pd.concat(sample_frames, ignore_index=True).to_csv(
            output_dir / "rolling_horizon_samples.csv", index=False
        )

    result = {
        "checkpoint": str(checkpoint),
        "model": args.model,
        "data_interval": "1min",
        "release_interval_minutes": interval,
        "forecast_horizon_minutes": pred_len,
        "publication_count": len(releases),
        "release_coverage": [
            str(dataset.timestamps[releases[0]]),
            str(dataset.timestamps[releases[-1]]),
        ],
        "all_60_forecast_minutes": all_horizon_metrics.result(),
        "minute_60_endpoint": endpoint_metrics.result(),
        "metric_scope": (
            "All 16 turbines combined. Acc30 uses truth > 100 kW; MAE uses all "
            "finite power points. Every input window ends at a real observation."
        ),
        "sample_count": int(model.n_sample),
    }
    _plot_endpoint_curve(
        endpoint_frame,
        result["minute_60_endpoint"],
        output_dir / "rolling_endpoint_wind_farm.png",
        args.model,
    )
    _plot_endpoint_diagnostics(endpoint_frame, output_dir, args.model)
    _plot_horizon_examples(
        sample_frames, output_dir / "rolling_horizon_examples.png"
    )
    _plot_training_history(checkpoint, output_dir)
    from exp.exp_wind_report import create_wind_report

    create_wind_report(
        prediction_curves, truth_curves, source_layout="NHT",
        dataset=dataset, output_dir=output_dir, figure_dir=output_dir,
        anchor_indices=releases, model_name=args.model, checkpoint=checkpoint,
        seed=args.seed, evaluation_name="rolling_15min",
    )
    result["layered_metrics_file"] = str(output_dir / "metrics.json")
    with open(output_dir / "rolling_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


class Exp_Wind_Rolling(Exp_Long_Term_Forecast):
    """Rolling publication-point evaluation for Guangning wind power."""

    def test(self) -> dict:
        return run_rolling_test(self.args, self.model_class)

