import numpy as np

from ..filters import ocmt_threshold


def ocmt_select(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    delta_penalise: float = 1.0,
    forecast_weights: np.ndarray | None = None,
) -> dict:
    """OCMT Stage-1 selection (Chudik et al. 2018 / 2024).

    For each j, regress Y on X_j (demeaned), threshold |t_j| against the
    Bonferroni critical value c_p(T-1, delta_penalise).  Selection is always
    full-sample and unweighted.

    forecast_weights (T,): if provided, uses WLS for the one-step-ahead
    forecast — this is the down-weighted OCMT variant (Chudik et al. 2024).

    Returns a MethodResult dict with K = 1, compatible with evaluation.py.
    """
    T, p = X.shape
    X_sel, Y_sel = X[:-1], Y[:-1]
    n = T - 1

    X_dm = X_sel - X_sel.mean(axis=0)
    Y_dm = Y_sel - Y_sel.mean()
    XX = np.sum(X_dm ** 2, axis=0)
    beta = (X_dm.T @ Y_dm) / np.where(XX > 0, XX, 1.0)
    ss_res = np.sum((Y_dm[:, None] - X_dm * beta) ** 2, axis=0)
    se = np.sqrt(ss_res / ((n - 2) * np.where(XX > 0, XX, 1.0)))
    t_stats = np.abs(beta / np.where(se > 0, se, np.inf))

    sel = t_stats >= ocmt_threshold(p, alpha, delta_penalise)
    sel_freq = sel.astype(float)

    if sel.sum() > 0:
        X_tr, Y_tr = X[:-1][:, sel], Y[:-1]
        if forecast_weights is not None:
            w = forecast_weights[:-1]
            coeffs, *_ = np.linalg.lstsq(X_tr * w[:, None], Y_tr * w, rcond=None)
        else:
            coeffs, *_ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
        forecast = float(X[-1, sel] @ coeffs)
    else:
        forecast = float(Y[:-1].mean())

    return {
        "selections":    [sel],
        "sel_freq":      sel_freq,
        "sel_majority":  sel,
        "ols_forecasts": [forecast],
    }
