import numpy as np
from sklearn import metrics as sk

from src.evaluation import mae, per_bin_metrics, r2, rmse


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 21, 500)
    y_pred = y_true + rng.normal(0, 2, 500)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(sk.mean_squared_error(y_true, y_pred)))
    assert np.isclose(mae(y_true, y_pred), sk.mean_absolute_error(y_true, y_pred))
    assert np.isclose(r2(y_true, y_pred), sk.r2_score(y_true, y_pred))


def test_per_bin_metrics_reports_bias_per_bin():
    y_true = np.array([1.0, 2.0, 19.0, 20.0])
    y_pred = np.array([2.0, 3.0, 18.0, 19.0])  # +1 on the low bin, -1 on the high bin
    table = per_bin_metrics(y_true, y_pred)
    assert table.loc["0-3", "bias"] == 1.0 and table.loc["18-21", "bias"] == -1.0
    assert table.loc["0-3", "n"] == 2 and table.loc["6-9", "n"] == 0
