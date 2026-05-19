import numpy as np

from ..knockoffs import construct_knockoffs
from ..statistics import select_lambda
from . import _knockoff_loop


def ipad_rolling(
    X: np.ndarray,
    Y: np.ndarray,
    q: float,
    h: int,
    B_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    s_val: np.ndarray,
    n_knockoffs: int = 100,
    rng: np.random.Generator = None,
) -> dict:
    """Rolling-window IPAD — LCD statistics restricted to the last h observations.

    Knockoffs are constructed from the full sample (P_X is assumed stable).
    Only the Lasso step is localised to the trailing window.
    """
    T = X.shape[0]
    mask = np.zeros(T, dtype=bool)
    mask[-(h + 1):-1] = True  # last h training rows; excludes test row Y[-1]
    X_ref = construct_knockoffs(X, B_hat, sigma_sq_hat, s_val, rng)
    lam = select_lambda(X, X_ref, Y, row_mask=mask)
    return _knockoff_loop(
        X, Y, B_hat, sigma_sq_hat, s_val, q, lam, n_knockoffs, rng,
        row_mask=mask,
    )
