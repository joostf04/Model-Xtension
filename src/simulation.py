import os
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from .dgp import generate_dataset
from .knockoffs import estimate_num_factors, estimate_factor_model, solve_sdp
from .methods.ipad import ipad_standard
from .methods.ipad_disc import ipad_discounted
from .methods.ipad_rolling import ipad_rolling
from .methods.oracle import ipad_oracle
from .methods.ocmt import ocmt_select
from .evaluation import evaluate_all


def build_grid() -> list[dict]:
    return [
        {"tau": tau, "overlap": overlap, "snr": snr, "delta": delta}
        for tau, overlap, snr, delta in product(
            config.TAU_GRID, config.OVERLAP_GRID, config.SNR_GRID, config.DELTA_GRID
        )
    ]


def run_replication(params: dict, seed: int) -> list[dict]:
    """All 6 methods for one Monte Carlo draw.  Returns a list of metric dicts."""
    rng = np.random.default_rng(seed)
    data = generate_dataset(
        config.T, config.P, config.S, config.R_TRUE,
        params["tau"], params["overlap"], params["snr"], rng,
    )
    X, Y = data["X"], data["Y"]
    beta_pre, beta_post, t_star = data["beta_pre"], data["beta_post"], data["t_star"]

    # Shared pre-computation — SDP solved once per replication for all IPAD variants
    r_hat = estimate_num_factors(X, config.R_MAX)
    B_hat, sigma_sq_hat = estimate_factor_model(X, r_hat)
    s_val = solve_sdp(B_hat @ B_hat.T + np.diag(sigma_sq_hat))

    delta = params["delta"]
    sw_disc = delta ** np.arange(config.T - 1, -1, -1)
    h = round(config.H_FRAC * config.T)

    ipad_kw = dict(
        B_hat=B_hat, sigma_sq_hat=sigma_sq_hat, s_val=s_val,
        n_knockoffs=config.N_KNOCKOFFS, rng=rng,
    )
    method_results = {
        "IPAD":         ipad_standard(X, Y, config.Q, **ipad_kw),
        "IPAD-disc":    ipad_discounted(X, Y, config.Q, delta, **ipad_kw),
        "IPAD-roll":    ipad_rolling(X, Y, config.Q, h, **ipad_kw),
        "IPAD-oracle":  ipad_oracle(X, Y, config.Q, t_star, **ipad_kw),
        "OCMT":         ocmt_select(X, Y, config.ALPHA_OCMT, config.DELTA_OCMT),
        "OCMT-dw":      ocmt_select(X, Y, config.ALPHA_OCMT, config.DELTA_OCMT,
                                    forecast_weights=sw_disc),
    }
    return [
        {"rep": seed, "method": method, **params,
         **evaluate_all(result, beta_pre, beta_post, t_star, X, Y)}
        for method, result in method_results.items()
    ]


def _replication_rolling_sensitivity(params: dict, seed: int) -> list[dict]:
    """Single replication for the rolling-window sensitivity appendix.

    Runs ipad_rolling with every h in config.H_GRID for one dataset draw.
    Uses the same shared pre-computation (SDP once per replication) as the
    main simulation.
    """
    rng = np.random.default_rng(seed)
    data = generate_dataset(
        config.T, config.P, config.S, config.R_TRUE,
        params["tau"], params["overlap"], params["snr"], rng,
    )
    X, Y = data["X"], data["Y"]
    beta_pre, beta_post, t_star = data["beta_pre"], data["beta_post"], data["t_star"]

    r_hat = estimate_num_factors(X, config.R_MAX)
    B_hat, sigma_sq_hat = estimate_factor_model(X, r_hat)
    s_val = solve_sdp(B_hat @ B_hat.T + np.diag(sigma_sq_hat))
    ipad_kw = dict(B_hat=B_hat, sigma_sq_hat=sigma_sq_hat, s_val=s_val,
                   n_knockoffs=config.N_KNOCKOFFS, rng=rng)

    rows = []
    for h_frac in config.H_GRID:
        h = round(h_frac * config.T)
        result = ipad_rolling(X, Y, config.Q, h, **ipad_kw)
        rows.append({
            "rep": seed, "method": f"IPAD-roll-h{h_frac}", "h_frac": h_frac,
            **params,
            **evaluate_all(result, beta_pre, beta_post, t_star, X, Y),
        })
    return rows


def run_rolling_sensitivity(
    cells: list[dict] | None = None,
    M: int | None = None,
    n_jobs: int = -1,
    output_path: str = "results/rolling_sensitivity.parquet",
) -> pd.DataFrame:
    """Rolling-window sensitivity appendix: run all h in H_GRID for a set of cells.

    Suggested cells: one representative tau × overlap × snr slice at fixed
    delta=1.00 (so that the only variable is h).  Defaults to the full grid
    at delta=1.00 if cells is None.
    """
    if cells is None:
        cells = [p for p in build_grid() if p["delta"] == 1.00]
    if M is None:
        M = config.M

    all_rows = Parallel(n_jobs=n_jobs)(
        delayed(_replication_rolling_sensitivity)(params, seed)
        for params in cells
        for seed in range(M)
    )
    df = pd.DataFrame([row for rep in all_rows for row in rep])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def run_cell(params: dict, M: int | None = None, n_jobs: int = -1) -> pd.DataFrame:
    if M is None:
        M = config.M
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_replication)(params, seed) for seed in range(M)
    )
    return pd.DataFrame([row for rep in rows for row in rep])


def run_simulation(
    grid: list[dict] | None = None,
    M: int | None = None,
    n_jobs: int = -1,
    checkpoint_dir: str = "results/checkpoints",
    output_path: str = "results/raw.parquet",
) -> pd.DataFrame:
    if grid is None:
        grid = build_grid()
    if M is None:
        M = config.M
    os.makedirs(checkpoint_dir, exist_ok=True)

    dfs = []
    for i, params in enumerate(grid):
        ckpt = os.path.join(checkpoint_dir, f"cell_{i:03d}.parquet")
        if os.path.exists(ckpt):
            dfs.append(pd.read_parquet(ckpt))
            print(f"[{i+1:3d}/{len(grid)}] loaded  {params}")
            continue
        df = run_cell(params, M, n_jobs)
        df.to_parquet(ckpt, index=False)
        dfs.append(df)
        print(f"[{i+1:3d}/{len(grid)}] done    {params}")

    result = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result
