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
    draws.  Re-running CV per draw (as Fan et al. 2020 do in their empirical
    application) would be 100x more expensive and has negligible impact on the
    simulation because the data matrix is fixed — only the knockoff columns vary.

    sample_weight: row-scaling factors sqrt(delta^(T-t)).  Applied as
    direct rescaling (Z *= w, Y *= w); because Lasso squares residuals the
    effective objective weight per observation is w_t^2 = delta^(T-t),
    matching the proposal's sum_t delta^(T-t) * residual_t^2.  sklearn's
    sample_weight is NOT used because it normalises weights to sum to
    n_samples, shifting the effective lambda.
    """
    Z, _Y = np.hstack([X, X_tilde]), Y
    if row_mask is not None:
        Z, _Y = Z[row_mask], Y[row_mask]
        sample_weight = sample_weight[row_mask] if sample_weight is not None else None
    if sample_weight is not None:
        Z, _Y = Z * sample_weight[:, None], _Y * sample_weight
    cv = LassoCV(cv=5, fit_intercept=False, max_iter=30_000, tol=1e-3, n_jobs=1)
    cv.fit(Z, _Y)
    return float(cv.alpha_)


def lcd_statistics(
    X: np.ndarray,
    X_tilde: np.ndarray,
    Y: np.ndarray,
    lam: float,
    sample_weight: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> np.ndarray:
    """W_j = |beta_j| - |beta_{j+p}| from Lasso on [X, X_tilde] at fixed lambda.

    sample_weight: see select_lambda for the rescaling convention.
    """
    p = X.shape[1]
    Z, _Y = np.hstack([X, X_tilde]), Y
    if row_mask is not None:
        Z, _Y = Z[row_mask], Y[row_mask]
        sample_weight = sample_weight[row_mask] if sample_weight is not None else None
    if sample_weight is not None:
        Z, _Y = Z * sample_weight[:, None], _Y * sample_weight
    model = Lasso(alpha=lam, fit_intercept=False, max_iter=30_000, tol=1e-3)
    model.fit(Z, _Y)
    beta = model.coef_
    return np.abs(beta[:p]) - np.abs(beta[p:])
