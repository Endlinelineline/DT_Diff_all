from typing import Optional, Union

import numpy as np


def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Multi-quantile (pinball) loss.

    Discretized CRPS via a left-Riemann sum over uniformly spaced quantiles.
    """
    if weights is None:
        weights = np.ones(y.shape)

    n_q = len(quantiles)
    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    loss = quantiles * sq + (1 - quantiles) * s1_q

    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    return np.average(loss, weights=weights, axis=axis)


def mae(y: np.ndarray, y_hat: np.ndarray, axis: Optional[int] = None):
    return np.nanmean(np.abs(y - y_hat), axis=axis)


def mse(y: np.ndarray, y_hat: np.ndarray, axis: Optional[int] = None):
    return np.nanmean(np.square(y - y_hat), axis=axis)


def calculate_metrics(y_pred, y_real, quantiles=(np.arange(9) + 1) / 10):
    """Return (MAE, MSE, MQL) plus the quantile / point predictions used."""
    y_pred = y_pred.cpu().numpy()
    y_real = y_real.cpu().numpy()

    assert y_pred.ndim == 4
    assert y_pred[0].shape == y_real.shape

    y_pred_point = np.mean(y_pred, axis=0)
    y_pred_q = np.quantile(y_pred, quantiles, axis=0)
    y_pred_q = np.transpose(y_pred_q, (1, 2, 3, 0))

    MAE = mae(y_real, y_pred_point)
    MSE = mse(y_real, y_pred_point)
    MQL = mqloss(y_real, y_pred_q, quantiles=np.array(quantiles))
    return (MAE, MSE, MQL), y_pred_q, y_pred_point


def tslib_reference_metrics(pred, true, chunk_rows=256):
    """Match Time-Series-Library metrics.py on [sample, horizon, channel] data.

    Arrays may be memory maps. CORR intentionally follows the supplied file's
    non-Pearson denominator and returns one value per horizon for 3-D inputs.
    """
    if pred.shape != true.shape or pred.ndim not in (2, 3) or not len(pred):
        raise ValueError("Prediction and truth need matching nonempty 2-D/3-D shapes")

    count = 0
    zero_count = 0
    nonzero_count = 0
    absolute_sum = squared_sum = true_sum = true_squared_sum = 0.0
    percentage_abs_sum = percentage_squared_sum = 0.0
    position_shape = pred.shape[1:]
    true_position_sum = np.zeros(position_shape, dtype=np.float64)
    pred_position_sum = np.zeros(position_shape, dtype=np.float64)

    for left in range(0, len(pred), chunk_rows):
        right = min(left + chunk_rows, len(pred))
        p = np.asarray(pred[left:right], dtype=np.float64)
        t = np.asarray(true[left:right], dtype=np.float64)
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            raise ValueError("Time-Series-Library metrics require finite arrays")
        error = t - p
        absolute_sum += float(np.abs(error).sum())
        squared_sum += float(np.square(error).sum())
        true_sum += float(t.sum())
        true_squared_sum += float(np.square(t).sum())
        count += t.size
        nonzero = t != 0
        zero_count += int(np.count_nonzero(~nonzero))
        nonzero_count += int(np.count_nonzero(nonzero))
        ratio = error[nonzero] / t[nonzero]
        percentage_abs_sum += float(np.abs(ratio).sum())
        percentage_squared_sum += float(np.square(ratio).sum())
        true_position_sum += t.sum(axis=0)
        pred_position_sum += p.sum(axis=0)

    true_mean = true_position_sum / len(pred)
    pred_mean = pred_position_sum / len(pred)
    numerator = np.zeros(position_shape, dtype=np.float64)
    denominator_squared = np.zeros(position_shape, dtype=np.float64)
    for left in range(0, len(pred), chunk_rows):
        right = min(left + chunk_rows, len(pred))
        centered_true = np.asarray(true[left:right], dtype=np.float64) - true_mean
        centered_pred = np.asarray(pred[left:right], dtype=np.float64) - pred_mean
        numerator += (centered_true * centered_pred).sum(axis=0)
        denominator_squared += (
            np.square(centered_true) * np.square(centered_pred)
        ).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (numerator / np.sqrt(denominator_squared)).mean(axis=-1)
    corr_array = np.asarray(corr)
    if corr_array.ndim == 0:
        corr_output = float(corr_array) if np.isfinite(corr_array) else None
    else:
        corr_output = [float(v) if np.isfinite(v) else None for v in corr_array]

    mse_value = squared_sum / count
    total_variance = true_squared_sum - true_sum**2 / count
    rse = np.sqrt(squared_sum / total_variance) if total_variance > 0 else None
    return {
        "MAE": absolute_sum / count,
        "MSE": mse_value,
        "RMSE": float(np.sqrt(mse_value)),
        "MAPE": percentage_abs_sum / count if not zero_count else None,
        "MSPE": percentage_squared_sum / count if not zero_count else None,
        "RSE": float(rse) if rse is not None else None,
        "CORR": corr_output,
        "evaluated_points": count,
        "zero_truth_points": zero_count,
        "MAPE_nonzero_truth_only": (
            percentage_abs_sum / nonzero_count if nonzero_count else None
        ),
        "MSPE_nonzero_truth_only": (
            percentage_squared_sum / nonzero_count if nonzero_count else None
        ),
        "notes": {
            "MAPE_MSPE": (
                "Original formulas divide by truth and are undefined when truth is zero; "
                "nonzero-only values are supplemental, not the original metric."
            ),
            "CORR": (
                "The supplied formula is not standard Pearson correlation; "
                "for 3-D input it returns one value per forecast horizon."
            ),
        },
    }
