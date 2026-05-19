import numpy as np


def fdr(result: dict, null_true: np.ndarray) -> float:
    """Mean FDP across K selections.  FDP = 0 when the selection is empty."""
    return float(np.mean([
        np.sum(sel & null_true) / max(1, int(sel.sum()))
        for sel in result["selections"]
    ]))


def power(result: dict, alt_true: np.ndarray) -> float:
    """Mean TPR across K selections.  NaN when there are no true alternatives."""
    n_alt = int(alt_true.sum())
    if n_alt == 0:
        return float("nan")
    return float(np.mean([
        np.sum(sel & alt_true) / n_alt for sel in result["selections"]
    ]))


def r2_oos(Y_last: float, Y_bar: float, ols_forecasts: list[float]) -> float:
    """Campbell-Thompson OOS R² with forecast averaged over K knockoff draws."""
    ss_bench = (Y_last - Y_bar) ** 2
    if ss_bench == 0.0:
        return float("nan")
    return float(1.0 - (Y_last - float(np.mean(ols_forecasts))) ** 2 / ss_bench)


def evaluate_all(
    result: dict,
    beta_pre: np.ndarray,
    beta_post: np.ndarray,
    t_star: int,
    X: np.ndarray,
    Y: np.ndarray,
) -> dict:
    """All four metrics for one (method, replication) pair.

    Null sets
    ---------
    H0^(2)  : beta_post_j = 0   (forecasting-relevant / local null)
    H0^avg  : T^{-1}(t_star * beta_pre + (T - t_star) * beta_post)_j = 0
    """
    T = X.shape[0]
    null_local = beta_post == 0.0
    null_avg = ((t_star * beta_pre + (T - t_star) * beta_post) / T) == 0.0

    return {
        "fdr_local":       fdr(result, null_local),
        "fdr_avg":         fdr(result, null_avg),
        "power_local":     power(result, ~null_local),
        "r2_oos":          r2_oos(float(Y[-1]), float(Y[:-1].mean()), result["ols_forecasts"]),
        "avg_discoveries": float(np.mean([sel.sum() for sel in result["selections"]])),
    }
