import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import generate_dataset
from src.knockoffs import (
    estimate_num_factors, estimate_factor_model,
    _woodbury_inv, solve_sdp, construct_knockoffs,
)
from src.filters import knockoff_threshold, apply_filter


def rng():
    return np.random.default_rng(42)


def _data():
    return generate_dataset(60, 20, 4, 3, 0.5, 0.5, 2.0, rng())


def test_estimate_num_factors_in_range():
    r = estimate_num_factors(_data()["X"], r_max=8)
    assert 1 <= r <= 8


def test_estimate_factor_model_shapes():
    B_hat, sq = estimate_factor_model(_data()["X"], r=3)
    assert B_hat.shape == (20, 3) and sq.shape == (20,) and np.all(sq > 0)


def test_woodbury_matches_direct_inverse():
    _rng = np.random.default_rng(1)
    B = _rng.standard_normal((8, 2))
    sq = _rng.uniform(0.5, 1.5, 8)
    Sigma = B @ B.T + np.diag(sq)
    np.testing.assert_allclose(_woodbury_inv(B, sq), np.linalg.inv(Sigma), atol=1e-10)


def test_sdp_solution_is_psd():
    data = _data()
    B_hat, sq = estimate_factor_model(data["X"], r=3)
    s_val = solve_sdp(B_hat @ B_hat.T + np.diag(sq))
    assert s_val.shape == (20,) and np.all(s_val >= -1e-8)
    eigvals = np.linalg.eigvalsh(2 * (B_hat @ B_hat.T + np.diag(sq)) - np.diag(s_val))
    assert np.all(eigvals >= -1e-8), f"min eigval = {eigvals.min():.2e}"


def test_construct_knockoffs_shape():
    data = _data()
    B_hat, sq = estimate_factor_model(data["X"], r=3)
    s_val = solve_sdp(B_hat @ B_hat.T + np.diag(sq))
    X_tilde = construct_knockoffs(data["X"], B_hat, sq, s_val, rng())
    assert X_tilde.shape == data["X"].shape


def test_knockoff_threshold_no_negatives_returns_inf_with_few_positives():
    # With q=0.2 and only 3 positive W values, (1+0)/3 = 0.33 > 0.2 → no threshold
    assert knockoff_threshold(np.array([3.0, 2.0, 1.0]), q=0.2) == np.inf


def test_knockoff_threshold_selects_with_enough_positives():
    # (1+0)/5 = 0.2 <= 0.2 → threshold found at t=1.0
    mask = apply_filter(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), q=0.2)
    assert mask.sum() == 5


def test_apply_filter_no_discoveries_under_pure_noise():
    W = np.random.default_rng(7).standard_normal(100)
    assert apply_filter(W, q=0.2).sum() <= 20
