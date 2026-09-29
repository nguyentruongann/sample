import numpy as np

from demand_forecasting.evaluate import calculate_metrics, wmape


def test_wmape_known_value():
    actual = np.array([10.0, 20.0, 30.0])
    prediction = np.array([8.0, 22.0, 27.0])
    assert np.isclose(wmape(actual, prediction), 100.0 * 7.0 / 60.0)


def test_metrics_ignore_non_finite_pairs():
    metrics = calculate_metrics([10.0, np.nan, 20.0], [8.0, 99.0, 22.0])
    assert metrics["rows"] == 2
    assert np.isclose(metrics["MAE"], 2.0)
    assert np.isclose(metrics["bias_%"], 0.0)
