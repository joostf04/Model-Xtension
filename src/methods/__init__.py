"""Shared one-step-ahead OLS helper and the core knockoff loop.

All four IPAD variants delegate to _knockoff_loop; the only differences
between them are the sample_weight and row_mask arguments.
"""
import numpy as np

from ..knockoffs import construct_knockoffs
from ..statistics import lcd_statistics
from ..filters import apply_filter


def _ols_forecast(
    X: np.ndarray,
    Y: np.ndarray,
    sel: np.ndarray,
    forecast_weights: np.ndarray | None = None,
) -> float:
    """Fit OLS on X[:-1][sel], Y[:-1] and predict Y[-1].

    Falls back to the historical mean when sel is empty.
    forecast_weights (T,) enables WLS; only the first T-1 entries are used.
    """
    if sel.sum() == 0:
        return float(Y[:-1].mean())
    X_tr = X[:-1][:, sel]
    Y_tr = Y[:-1]
    if forecast_weights is not None:
        w = forecast_weights[:-1]
        coeffs, *_ = np.linalg.lstsq(X_tr * w[:, None], Y_tr * w, rcond=None)
    else:
        coeffs, *_ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
    return float(X[-1, sel] @ coeffs)


def _knockoff_loop(
    X: np.ndarray,
    Y: np.ndarray,
    B_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    s_val: np.ndarray,
    q: float,
    lam: float,
    n_knockoffs: int,
    rng: np.random.Generator,
    sample_weight: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> dict:
    """Run n_knockoffs knockoff draws at fixed lambda; return a MethodResult.

    MethodResult schema
    -------------------
    selections    : list[np.ndarray]  K boolean masks (p,)
    sel_freq      : np.ndarray        (p,) mean selection rate across K draws
    sel_majority  : np.ndarray        (p,) bool, sel_freq >= 0.5
    ols_forecasts : list[float]       K one-step-ahead OLS predictions of Y[-1]
    """
    selections: list[np.ndarray] = []
    ols_forecasts: list[float] = []

    for _ in range(n_knockoffs):
        X_tilde = construct_knockoffs(X, B_hat, sigma_sq_hat, s_val, rng)
        W = lcd_statistics(X, X_tilde, Y, lam, sample_weight, row_mask)
        sel = apply_filter(W, q)
        selections.append(sel)
        ols_forecasts.append(_ols_forecast(X, Y, sel))

    sel_freq = np.mean(selections, axis=0)
    return {
        "selections":    selections,
        "sel_freq":      sel_freq,
        "sel_majority":  sel_freq >= 0.5,
        "ols_forecasts": ols_forecasts,
    }
