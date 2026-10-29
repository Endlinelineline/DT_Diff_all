"""Unified wind-power metrics for all Guangning turbines."""

from __future__ import annotations

import numpy as np


class WindMetricAccumulator:
    """Accumulate MAE and Acc30 without retaining every forecast in memory.

    Acc30 follows the project convention: among points whose observed power is
    greater than 100 kW, a point is correct when its absolute error is no more
    than 30 percent of the observed power. MAE uses every finite point.
    """

    def __init__(self) -> None:
        self.absolute_error_sum = 0.0
        self.point_count = 0
        self.acc30_passed = 0
        self.acc30_valid = 0

    def update(self, prediction, truth) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        if prediction.shape != truth.shape:
            raise ValueError(
                f"Prediction/truth shape mismatch: {prediction.shape} vs {truth.shape}"
            )
        finite = np.isfinite(prediction) & np.isfinite(truth)
        if not finite.all():
            prediction = prediction[finite]
            truth = truth[finite]
        error = np.abs(prediction - truth)
        self.absolute_error_sum += float(error.sum())
        self.point_count += int(error.size)
        valid = truth > 100.0
        self.acc30_valid += int(np.count_nonzero(valid))
        self.acc30_passed += int(
            np.count_nonzero(valid & (error <= 0.30 * truth))
        )

    def result(self) -> dict:
        if self.point_count == 0:
            raise ValueError("No finite points were available for metric calculation")
        acc30 = None
        if self.acc30_valid:
            acc30 = 100.0 * self.acc30_passed / self.acc30_valid
        return {
            "Acc30": acc30,
            "MAE_kW": self.absolute_error_sum / self.point_count,
            "evaluated_points": self.point_count,
            "acc30_valid_points": self.acc30_valid,
            "acc30_passed_points": self.acc30_passed,
        }


def to_sample_turbine_horizon(values, source_layout: str) -> np.ndarray:
    """Return the evaluation layout [sample, turbine, horizon].

    Model output is [sample, horizon, turbine]; conversion belongs at the
    evaluation boundary, never inside the forecasting model.
    """
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D forecast array, got {array.shape}")
    if source_layout == "NTH":
        return array
    if source_layout == "NHT":
        return np.swapaxes(array, 1, 2)
    raise ValueError("source_layout must be 'NTH' or 'NHT'")


_METRIC_KEYS = (
    "strict_acc30_pct", "MAE_kW", "MSE_kW2", "RMSE_kW", "MAPE_pct"
)


def _summarize(total, finite, valid, passed, abs_sum, sq_sum, ape_sum) -> dict:
    total, finite, valid, passed = map(int, (total, finite, valid, passed))
    mse = float(sq_sum / finite) if finite else None
    return {
        "strict_acc30_pct": float(100 * passed / valid) if valid else None,
        "MAE_kW": float(abs_sum / finite) if finite else None,
        "MSE_kW2": mse,
        "RMSE_kW": float(np.sqrt(mse)) if mse is not None else None,
        "MAPE_pct": float(100 * ape_sum / valid) if valid else None,
        "valid_power_points": valid,
        "total_points": total,
        "valid_power_ratio_pct": float(100 * valid / total) if total else None,
        "finite_points": finite,
        "acc30_passed_points": passed,
    }


def _macro(rows: list[dict]) -> dict:
    result = {}
    for key in _METRIC_KEYS:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[f"{key}_turbine_count"] = len(values)
    for key in ("valid_power_points", "total_points", "finite_points", "acc30_passed_points"):
        result[key] = sum(row[key] for row in rows)
    total = result["total_points"]
    result["valid_power_ratio_pct"] = (
        100 * result["valid_power_points"] / total if total else None
    )
    return result


class LayeredWindMetrics:
    """Streaming kW-space metrics for canonical [N, turbine, horizon] arrays."""

    def __init__(self, turbine_ids: list[str], prediction_steps: int):
        if not turbine_ids or prediction_steps < 1:
            raise ValueError("At least one turbine and one prediction step are required")
        self.turbine_ids = list(turbine_ids)
        self.prediction_steps = int(prediction_steps)
        shape = (len(turbine_ids), self.prediction_steps)
        self.total = np.zeros(shape, dtype=np.int64)
        self.finite = np.zeros(shape, dtype=np.int64)
        self.valid = np.zeros(shape, dtype=np.int64)
        self.passed = np.zeros(shape, dtype=np.int64)
        self.abs_sum = np.zeros(shape, dtype=np.float64)
        self.sq_sum = np.zeros(shape, dtype=np.float64)
        self.ape_sum = np.zeros(shape, dtype=np.float64)
        self.site_total = np.zeros(self.prediction_steps, dtype=np.int64)
        self.site_finite = np.zeros(self.prediction_steps, dtype=np.int64)
        self.site_valid = np.zeros(self.prediction_steps, dtype=np.int64)
        self.site_passed = np.zeros(self.prediction_steps, dtype=np.int64)
        self.site_abs_sum = np.zeros(self.prediction_steps, dtype=np.float64)
        self.site_sq_sum = np.zeros(self.prediction_steps, dtype=np.float64)
        self.site_ape_sum = np.zeros(self.prediction_steps, dtype=np.float64)
        self.sample_count = 0

    def update(self, prediction_kw, truth_kw) -> None:
        prediction = np.asarray(prediction_kw, dtype=np.float64)
        truth = np.asarray(truth_kw, dtype=np.float64)
        expected = (len(self.turbine_ids), self.prediction_steps)
        if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[1:] != expected:
            raise ValueError(
                f"Expected matching [N,{expected[0]},{expected[1]}] arrays; "
                f"got {prediction.shape} and {truth.shape}"
            )
        n = prediction.shape[0]
        self.sample_count += n
        self.total += n
        finite = np.isfinite(prediction) & np.isfinite(truth)
        valid = finite & (truth > 100.0)
        error = np.where(finite, prediction - truth, 0.0)
        absolute = np.abs(error)
        self.finite += finite.sum(axis=0)
        self.valid += valid.sum(axis=0)
        self.passed += (valid & (absolute <= 0.30 * truth)).sum(axis=0)
        self.abs_sum += absolute.sum(axis=0)
        self.sq_sum += np.square(error).sum(axis=0)
        self.ape_sum += np.divide(
            absolute, truth, out=np.zeros_like(absolute), where=valid
        ).sum(axis=0)

        # A site point is evaluable only when all turbine values are finite.
        site_finite = finite.all(axis=1)
        site_prediction = np.where(finite, prediction, 0.0).sum(axis=1)
        site_truth = np.where(finite, truth, 0.0).sum(axis=1)
        site_valid = site_finite & (site_truth > 100.0)
        site_error = np.where(site_finite, site_prediction - site_truth, 0.0)
        site_absolute = np.abs(site_error)
        self.site_total += n
        self.site_finite += site_finite.sum(axis=0)
        self.site_valid += site_valid.sum(axis=0)
        self.site_passed += (site_valid & (site_absolute <= 0.30 * site_truth)).sum(axis=0)
        self.site_abs_sum += site_absolute.sum(axis=0)
        self.site_sq_sum += np.square(site_error).sum(axis=0)
        self.site_ape_sum += np.divide(
            site_absolute, site_truth, out=np.zeros_like(site_absolute), where=site_valid
        ).sum(axis=0)

    def _turbine_stats(self, selection) -> dict:
        return _summarize(
            self.total[selection].sum(), self.finite[selection].sum(),
            self.valid[selection].sum(), self.passed[selection].sum(),
            self.abs_sum[selection].sum(), self.sq_sum[selection].sum(),
            self.ape_sum[selection].sum(),
        )

    def _site_stats(self, selection) -> dict:
        return _summarize(
            self.site_total[selection].sum(), self.site_finite[selection].sum(),
            self.site_valid[selection].sum(), self.site_passed[selection].sum(),
            self.site_abs_sum[selection].sum(), self.site_sq_sum[selection].sum(),
            self.site_ape_sum[selection].sum(),
        )

    def result(self) -> dict:
        if self.sample_count == 0:
            raise ValueError("No samples were evaluated")
        per_turbine_curve = [
            {"turbine_id": name, **self._turbine_stats((index, slice(None)))}
            for index, name in enumerate(self.turbine_ids)
        ]
        per_turbine_endpoint = [
            {"turbine_id": name, **self._turbine_stats((index, -1))}
            for index, name in enumerate(self.turbine_ids)
        ]
        per_horizon = []
        for horizon in range(self.prediction_steps):
            per_turbine_at_horizon = [
                self._turbine_stats((index, horizon))
                for index in range(len(self.turbine_ids))
            ]
            per_horizon.append({
                "prediction_step": horizon + 1,
                "curve_overall": self._turbine_stats((slice(None), horizon)),
                "macro_turbine_curve": _macro(per_turbine_at_horizon),
                "site_overall": self._site_stats(horizon),
            })
        return {
            "shape": [self.sample_count, len(self.turbine_ids), self.prediction_steps],
            "units": {"power": "kW", "MSE": "kW^2", "Acc30": "percent", "MAPE": "percent"},
            "definition": "Strict Acc30 and MAPE use paired finite points with truth > 100 kW; other errors use all paired finite points. Site power is the turbine sum.",
            "curve_overall": self._turbine_stats((slice(None), slice(None))),
            "endpoint_overall": self._turbine_stats((slice(None), -1)),
            "per_turbine_curve": per_turbine_curve,
            "per_turbine_endpoint": per_turbine_endpoint,
            "macro_turbine_curve": _macro(per_turbine_curve),
            "macro_turbine_endpoint": _macro(per_turbine_endpoint),
            "site_overall": {
                "curve": self._site_stats(slice(None)),
                "endpoint": self._site_stats(-1),
            },
            "per_horizon": per_horizon,
        }


def better_validation_score(candidate_acc30, candidate_mse, best_acc30, best_mse) -> bool:
    """Strictly lexicographic: higher Acc30, then lower MSE on an exact tie."""
    if candidate_acc30 is None:
        return False
    if best_acc30 is None:
        return True
    return candidate_acc30 > best_acc30 or (
        candidate_acc30 == best_acc30 and candidate_mse < best_mse
    )

