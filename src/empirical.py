"""Goyal-Welch equity premium OOS application (to be implemented).

Data
----
Monthly equity premium r_{t+1}^e = r_{t+1}^mkt - r_f.
14 Welch & Goyal (2008) predictors + 3 lags each → p = 56.
Sample: 1950:01 – 2023:12.  OOS evaluation: 1966:01 – 2023:12.

Procedure (expanding window)
-----------------------------
At each forecast origin t:
  1. Run IPAD, Discounted IPAD (delta chosen by 60-month MSE), and OCMT on
     the full history up to t.
  2. Produce one-step-ahead forecast of r_{t+1}^e.
  3. Record selected variables and forecast.

Output
------
- Campbell-Thompson R²_OOS vs. historical-mean benchmark.
- Selection frequency heat-maps over time (detect instability around 2008, COVID).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_goyal_welch(path: str = "data/raw/goyal_welch.csv") -> pd.DataFrame:
    """Load the Goyal-Welch CSV and return a tidy DataFrame.

    Expected columns: date (YYYYMM), equity_premium, and the 14 predictors.
    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    raise NotImplementedError(
        f"Place the Goyal-Welch CSV at {path} and implement this loader."
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(raw: pd.DataFrame, n_lags: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) with X containing predictors and their first n_lags lags.

    Returns
    -------
    X : (T, p) array,  p = 14 * (1 + n_lags)
    Y : (T,)   equity premium
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Delta selection (cross-validated, used in empirical section only)
# ---------------------------------------------------------------------------

def select_delta(
    X: np.ndarray,
    Y: np.ndarray,
    delta_grid: list[float],
    window: int = 60,
) -> float:
    """Choose delta by minimising MSE over the preceding `window` months."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main expanding-window loop
# ---------------------------------------------------------------------------

def run_empirical(
    X: np.ndarray,
    Y: np.ndarray,
    oos_start: int,
    delta_grid: list[float] | None = None,
    n_knockoffs: int = 100,
) -> pd.DataFrame:
    """Expanding-window OOS evaluation.

    Returns a DataFrame with columns: date, method, forecast, r2_oos,
    and selection indicators for each predictor.
    """
    raise NotImplementedError
