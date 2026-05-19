import numpy as np

from ..knockoffs import construct_knockoffs
from ..statistics import select_lambda
from . import _knockoff_loop


def ipad_oracle(
    X: np.ndarray,
    Y: np.ndarray,
    q: float,
    t_star: int,
    B_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    s_val: np.ndarray,
    n_knockoffs: int = 100,
    rng: np.random.Generator = None,
) -> dict:
    """Oracle IPAD — LCD statistics restricted to post-break observations only.

    Knows the true break date t_star; provides an upper bound on what any
    localised method can achieve without knowledge of break timing.
    """
    T = X.shape[0]
    mask = np.zeros(T, dtype=bool)
    mask[t_star:-1] = True  # post-break training rows; excludes test row Y[-1]
    X_ref = construct_knockoffs(X, B_hat, sigma_sq_hat, s_val, rng)
    lam = select_lambda(X, X_ref, Y, row_mask=mask)
    return _knockoff_loop(
        X, Y, B_hat, sigma_sq_hat, s_val, q, lam, n_knockoffs, rng,
        row_mask=mask,
    )
