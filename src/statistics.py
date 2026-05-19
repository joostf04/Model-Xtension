"""LCD W-statistics and cross-validated lambda selection."""
import numpy as np
from sklearn.linear_model import Lasso, LassoCV


def select_lambda(
    X: np.ndarray,
    X_tilde: np.ndarray,
    Y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> float:
    """Cross-validate lambda on one augmented [X, X_tilde] draw.

    Called once per (replication, method) and reused across all 100 knockoff
    draws so that CV is not run 100 times per method.
    """
    Z, _Y, _sw = np.hstack([X, X_tilde]), Y, sample_weight
    if row_mask is not None:
        Z, _Y = Z[row_mask], Y[row_mask]
        _sw = sample_weight[row_mask] if sample_weight is not None else None
    cv = LassoCV(cv=5, fit_intercept=False, max_iter=10_000, n_jobs=1)
    cv.fit(Z, _Y, sample_weight=_sw)
    return float(cv.alpha_)


def lcd_statistics(
    X: np.ndarray,
    X_tilde: np.ndarray,
    Y: np.ndarray,
    lam: float,
    sample_weight: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> np.ndarray:
    """W_j = |beta_j| - |beta_{j+p}| from Lasso on [X, X_tilde] at fixed lambda."""
    p = X.shape[1]
    Z, _Y, _sw = np.hstack([X, X_tilde]), Y, sample_weight
    if row_mask is not None:
        Z, _Y = Z[row_mask], Y[row_mask]
        _sw = sample_weight[row_mask] if sample_weight is not None else None
    model = Lasso(alpha=lam, fit_intercept=False, max_iter=10_000)
    model.fit(Z, _Y, sample_weight=_sw)
    beta = model.coef_
    return np.abs(beta[:p]) - np.abs(beta[p:])
