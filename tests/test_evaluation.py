import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import fdr, power, r2_oos, evaluate_all


def _result(selections, forecasts=None):
    sels = [np.asarray(s, dtype=bool) for s in selections]
    freq = np.mean(sels, axis=0)
    return {
        "selections":    sels,
        "sel_freq":      freq,
        "sel_majority":  freq >= 0.5,
        "ols_forecasts": forecasts if forecasts is not None else [0.0] * len(sels),
    }


def test_fdr_no_false_discoveries():
    null = np.array([False, False, True, True])
    assert fdr(_result([np.array([True, True, False, False])]), null) == 0.0


def test_fdr_all_false_discoveries():
    null = np.array([True, True, True, True])
    assert fdr(_result([np.array([True, True, False, False])]), null) == 1.0


def test_fdr_empty_selection_is_zero():
    assert fdr(_result([np.zeros(4, bool)]), np.ones(4, bool)) == 0.0


def test_power_perfect():
    alt = np.array([True, True, False])
    assert power(_result([np.array([True, True, False])]), alt) == 1.0


def test_power_zero():
    alt = np.array([True, True, False])
    assert power(_result([np.array([False, False, True])]), alt) == 0.0


def test_power_no_alternatives_is_nan():
    assert np.isnan(power(_result([np.array([True, False])]), np.zeros(2, bool)))


def test_r2_oos_perfect_forecast():
    assert r2_oos(1.0, 0.0, [1.0]) == 1.0


def test_r2_oos_benchmark_forecast():
    assert r2_oos(2.0, 0.0, [0.0]) == 0.0


def test_r2_oos_worse_than_benchmark():
    assert r2_oos(1.0, 0.0, [3.0]) < 0.0


def test_r2_oos_averages_over_k():
    # average of [0, 2] = 1 = Y_last  →  R² = 1
    assert r2_oos(1.0, 0.0, [0.0, 2.0]) == 1.0


def test_evaluate_all_returns_all_keys():
    rng = np.random.default_rng(0)
    T, p = 50, 10
    X = rng.standard_normal((T, p))
    Y = rng.standard_normal(T)
    beta_pre = np.zeros(p);  beta_pre[:3] = 1.0
    beta_post = np.zeros(p); beta_post[3:6] = 1.0
    sel = np.zeros(p, bool);  sel[:3] = True
    res = _result([sel], [float(Y[:-1].mean())])
    m = evaluate_all(res, beta_pre, beta_post, t_star=25, X=X, Y=Y)
    assert set(m) == {"fdr_local", "fdr_avg", "power_local", "r2_oos", "avg_discoveries"}
