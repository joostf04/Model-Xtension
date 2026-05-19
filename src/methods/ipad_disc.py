import numpy as np

from ..knockoffs import construct_knockoffs
from ..statistics import select_lambda
from . import _knockoff_loop


def ipad_discounted(
    X: np.ndarray,
    Y: np.ndarray,
    q: float,
    delta: float,
    B_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    s_val: np.ndarray,
    n_knockoffs: int = 100,
    rng: np.random.Generator = None,
) -> dict:
    """Discounted IPAD — exponential sample weights w_t = sqrt(delta^(T-1-t)).

    Observation t = T-1 receives scale factor 1; t = 0 receives delta^((T-1)/2).
    Because the row-scaled data are squared inside the Lasso, the effective
    objective weight per observation is w_t^2 = delta^(T-1-t), matching the
    proposal's weighted Lasso objective sum_t delta^(T-1-t) * residual_t^2.
    Effective sample size n_eff ≈ (1+delta)/(1-delta) for large T.
    At delta=1 reduces to standard IPAD.

    The Lasso selection uses rows 0..T-2 only (excludes test row Y[-1]).
    Knockoff construction uses all T rows of X.

    Design note — intentional asymmetry vs OCMT-dw
    ------------------------------------------------
    This method localises the *selection* step via a weighted Lasso.
    The post-selection OLS forecast is deliberately unweighted.  OCMT-dw
    does the opposite: unweighted selection, weighted forecast.  The two
    methods bracket the question of where localisation matters most.
    """
    T = X.shape[0]
    sw = np.sqrt(delta ** np.arange(T - 1, -1, -1))
    mask = np.zeros(T, dtype=bool)
    mask[:-1] = True
    X_ref = construct_knockoffs(X, B_hat, sigma_sq_hat, s_val, rng)
    lam = select_lambda(X, X_ref, Y, sample_weight=sw, row_mask=mask)
    return _knockoff_loop(
        X, Y, B_hat, sigma_sq_hat, s_val, q, lam, n_knockoffs, rng,
        sample_weight=sw, row_mask=mask,
    )
