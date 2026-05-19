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
    """Standard IPAD (Fan et al. 2020) — full-sample, equal-weighted Lasso.

    The Lasso selection uses rows 0..T-2 only (row_mask excludes the test row
    Y[-1] to keep the evaluation truly out-of-sample, consistent with OCMT).
    Knockoff construction uses all T rows of X so the factor model sees the
    full covariate distribution including the forecast-period covariates.
    """
    T = X.shape[0]
    mask = np.zeros(T, dtype=bool)
    mask[:-1] = True
    X_ref = construct_knockoffs(X, B_hat, sigma_sq_hat, s_val, rng)
    lam = select_lambda(X, X_ref, Y, row_mask=mask)
    return _knockoff_loop(X, Y, B_hat, sigma_sq_hat, s_val, q, lam, n_knockoffs, rng,
                          row_mask=mask)
