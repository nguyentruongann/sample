"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def _finite_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    valid = np.isfinite(actual) & np.isfinite(prediction)
    return actual[valid], prediction[valid]


def wmape(y_true, y_pred) -> float:
    actual, prediction = _finite_arrays(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float(100.0 * np.abs(actual - prediction).sum() / denominator)


def calculate_metrics(y_true, y_pred) -> dict[str, float | int]:
    actual, prediction = _finite_arrays(y_true, y_pred)
    if not len(actual):
        return {
            "rows": 0,
            "actual_volume": 0.0,
            "predicted_volume": 0.0,
            "WMAPE_%": float("nan"),
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "bias_%": float("nan"),
            "underprediction_rate_%": float("nan"),
            "P90_abs_error": float("nan"),
            "P95_abs_error": float("nan"),
        }

    error = prediction - actual
    absolute_error = np.abs(error)
    denominator = np.abs(actual).sum()
    return {
        "rows": len(actual),
        "actual_volume": float(actual.sum()),
        "predicted_volume": float(prediction.sum()),
        "WMAPE_%": float(100.0 * absolute_error.sum() / denominator)
        if denominator
        else float("nan"),
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias_%": float(100.0 * error.sum() / denominator) if denominator else float("nan"),
        "underprediction_rate_%": float(100.0 * np.mean(error < 0)),
        "P90_abs_error": float(np.quantile(absolute_error, 0.90)),
        "P95_abs_error": float(np.quantile(absolute_error, 0.95)),
    }
