"""Factor model estimation, SDP s-values, and Gaussian knockoff sampling."""
import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------------
# Factor model estimation
# ---------------------------------------------------------------------------

def estimate_num_factors(X: np.ndarray, r_max: int = 10) -> int:
    """Ahn-Horenstein (2013) eigenvalue ratio test."""
    T, p = X.shape
    eigvals = np.sort(np.linalg.eigvalsh(X.T @ X / T))[::-1]
    r_max = min(r_max, len(eigvals) - 1)
    denom = np.where(eigvals[1:r_max + 1] > 0, eigvals[1:r_max + 1], 1e-12)
    return int(np.argmax(eigvals[:r_max] / denom) + 1)


def estimate_factor_model(
    X: np.ndarray, r: int
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-based factor model.  Normalisation: F_hat'F_hat / T = I_r.

    Returns
    -------
    B_hat       : (p, r) loading matrix
    sigma_sq_hat: (p,)   idiosyncratic variances
    """
    T = X.shape[0]
    U, _, _ = np.linalg.svd(X / np.sqrt(T), full_matrices=False)
    F_hat = U[:, :r] * np.sqrt(T)
    B_hat = X.T @ F_hat / T
    sigma_sq_hat = np.mean((X - F_hat @ B_hat.T) ** 2, axis=0)
    return B_hat, sigma_sq_hat


# ---------------------------------------------------------------------------
# Knockoff construction
# ---------------------------------------------------------------------------

def _woodbury_inv(B: np.ndarray, sigma_sq: np.ndarray) -> np.ndarray:
    """(B B' + diag(sigma_sq))^{-1} via the Woodbury identity."""
    D_inv = 1.0 / sigma_sq
    DinvB = B * D_inv[:, None]
    return np.diag(D_inv) - DinvB @ np.linalg.solve(np.eye(B.shape[1]) + B.T @ DinvB, DinvB.T)


def solve_sdp(Sigma: np.ndarray) -> np.ndarray:
    """SDP s-values: max sum(s) s.t. 2*Sigma - diag(s) >= 0, s >= 0.

    Falls back to the equicorrelated construction if the solver fails.
    Post-processes the solution to guarantee 2*Sigma - diag(s) is PSD
    despite SCS's finite tolerance.
    """
    p = Sigma.shape[0]
    s = cp.Variable(p, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(s)), [2 * Sigma - cp.diag(s) >> 0])
    prob.solve(solver=cp.SCS, eps=1e-4, verbose=False)

    if s.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        lam_min = float(np.linalg.eigvalsh(Sigma).min())
        return np.full(p, min(2.0 * lam_min, 1.0))

    s_val = np.clip(s.value, 0.0, 2.0 * np.diag(Sigma))

    # SCS can violate the PSD constraint by ~eps; shift s_val down until M is PD.
    # Reducing each s_j by delta raises every eigenvalue of M by at least delta.
    lam_min = float(np.linalg.eigvalsh(2.0 * Sigma - np.diag(s_val)).min())
    if lam_min < 1e-10:
        s_val = np.maximum(0.0, s_val - (abs(lam_min) + 1e-10))

    return s_val


def construct_knockoffs(
    X: np.ndarray,
    B_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    s_val: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one set of second-order Gaussian knockoffs.

    X_tilde | X ~ N(mu, V) where
      mu = X (I - Sigma^{-1} diag(s))'
      V  = 2 diag(s) - diag(s) Sigma^{-1} diag(s)
    """
    T, p = X.shape
    Sigma_inv = _woodbury_inv(B_hat, sigma_sq_hat)
    mu = X @ (np.eye(p) - np.diag(s_val) @ Sigma_inv).T
    V = 2.0 * np.diag(s_val) - np.diag(s_val) @ Sigma_inv @ np.diag(s_val)
    V = (V + V.T) * 0.5 + 1e-10 * np.eye(p)
    return mu + rng.standard_normal((T, p)) @ np.linalg.cholesky(V).T
