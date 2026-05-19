import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import make_factor_model, simulate_X, make_beta, generate_dataset


def rng():
    return np.random.default_rng(0)


def test_simulate_X_shape():
    B, s = make_factor_model(20, 3, rng())
    assert simulate_X(50, B, s, rng()).shape == (50, 20)


def test_simulate_X_unit_l2_norm():
    B, s = make_factor_model(20, 3, rng())
    X = simulate_X(500, B, s, rng())
    np.testing.assert_allclose(np.linalg.norm(X, axis=0), 1.0, rtol=1e-10)


def test_make_beta_break_location():
    *_, t_star = make_beta(50, 5, 0.6, 0.5, 2.0, 100, rng())
    assert t_star == 60


def test_make_beta_zero_overlap():
    _, beta_pre, beta_post, _ = make_beta(200, 20, 0.5, 0.0, 2.0, 100, rng())
    S1 = set(np.where(beta_pre != 0)[0])
    S2 = set(np.where(beta_post != 0)[0])
    assert len(S1) == 20 and len(S2) == 20 and len(S1 & S2) == 0


def test_make_beta_full_overlap():
    _, beta_pre, beta_post, _ = make_beta(200, 20, 0.5, 1.0, 2.0, 100, rng())
    np.testing.assert_array_equal(beta_pre != 0, beta_post != 0)


def test_generate_dataset_shapes():
    data = generate_dataset(50, 20, 4, 3, 0.5, 0.5, 2.0, rng())
    assert data["X"].shape == (50, 20)
    assert data["Y"].shape == (50,)
    assert data["beta"].shape == (50, 20)
    assert data["beta_pre"].shape == (20,)
    assert data["beta_post"].shape == (20,)
