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
    """Discounted IPAD — exponential sample weights w_t = delta^(T-1-t).

    Observation t = T-1 receives weight 1; t = 0 receives delta^(T-1).
    At delta = 1 this reduces to standard IPAD (useful sanity check).

    Design note — intentional asymmetry vs OCMT-dw
    ------------------------------------------------
    This method localises the *selection* step via a weighted Lasso.
    The post-selection OLS forecast is deliberately unweighted: the goal is
    to isolate whether localised selection improves forecasting, independently
    of any localisation at the forecasting stage.  OCMT-dw does the opposite:
    unweighted selection, weighted forecast.  The two methods bracket the
    question of where localisation matters most.
    """
    T = X.shape[0]
    sw = delta ** np.arange(T - 1, -1, -1)
    X_ref = construct_knockoffs(X, B_hat, sigma_sq_hat, s_val, rng)
    lam = select_lambda(X, X_ref, Y, sample_weight=sw)
    return _knockoff_loop(
        X, Y, B_hat, sigma_sq_hat, s_val, q, lam, n_knockoffs, rng,
        sample_weight=sw,
    )
