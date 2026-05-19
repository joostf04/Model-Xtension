"""Knockoff+ filter and OCMT critical value."""
import numpy as np
from scipy import stats


def knockoff_threshold(W: np.ndarray, q: float) -> float:
    """Knockoff+ filter threshold T+.

    T+ = min{ t in |W| : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q }

    Returns np.inf when no threshold satisfies the inequality (select nothing).
    """
    candidates = np.sort(np.unique(np.abs(W[W != 0])))
    for t in candidates:
        if (1.0 + np.sum(W <= -t)) / max(1, np.sum(W >= t)) <= q:
            return float(t)
    return np.inf


def apply_filter(W: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of variables selected by the knockoff+ filter at level q."""
    return W >= knockoff_threshold(W, q)


def ocmt_threshold(p: int, T: int, alpha: float = 0.05, delta: float = 1.0) -> float:
    """OCMT critical value c_p(T, delta) = Phi^{-1}(1 - alpha / (2 * p^delta))."""
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * p ** delta)))
