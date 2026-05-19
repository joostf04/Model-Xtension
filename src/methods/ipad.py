import numpy as np

from ..knockoffs import construct_knockoffs
from ..statistics import select_lambda
from . import _knockoff_loop


def ipad_standard(
    X: np.ndarray,
    Y: np.ndarray,
    q: float,
    B_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    s_val: np.ndarray,
    n_knockoffs: int = 100,
    rng: np.random.Generator = None,
) -> dict:
    """Standard IPAD (Fan et al. 2020) — full-sample, equal-weighted Lasso."""
    X_ref = construct_knockoffs(X, B_hat, sigma_sq_hat, s_val, rng)
    lam = select_lambda(X, X_ref, Y)
    return _knockoff_loop(X, Y, B_hat, sigma_sq_hat, s_val, q, lam, n_knockoffs, rng)
