import numpy as np


def make_factor_model(
    p: int, r: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    B = rng.uniform(-1.0, 1.0, size=(p, r))
    sigma_sq = rng.uniform(0.5, 1.5, size=p)
    return B, sigma_sq


def simulate_X(
    T: int, B: np.ndarray, sigma_sq: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    p, r = B.shape[0], B.shape[1]
    F = rng.standard_normal((T, r))
    U = rng.standard_normal((T, p)) * np.sqrt(sigma_sq)
    X = F @ B.T + U
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    return X / norms


def make_beta(
    p: int, s: int, tau: float, overlap: float, snr: float, T: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    t_star = int(np.floor(tau * T))
    n_overlap = int(np.floor(overlap * s))

    S1 = rng.choice(p, size=s, replace=False)
    shared = rng.choice(S1, size=n_overlap, replace=False) if n_overlap > 0 else np.array([], dtype=int)
    complement = np.setdiff1d(np.arange(p), S1)
    new_vars = rng.choice(complement, size=s - n_overlap, replace=False)
    S2 = np.concatenate([shared, new_vars]).astype(int)

    coef = np.sqrt(snr * T / s)
    beta_pre = np.zeros(p)
    beta_pre[S1] = coef
    beta_post = np.zeros(p)
    beta_post[S2] = coef

    beta = np.empty((T, p))
    beta[:t_star] = beta_pre
    beta[t_star:] = beta_post

    return beta, beta_pre, beta_post, t_star


def simulate_Y(
    X: np.ndarray, beta: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return np.sum(X * beta, axis=1) + rng.standard_normal(X.shape[0])


def generate_dataset(
    T: int, p: int, s: int, r: int,
    tau: float, overlap: float, snr: float,
    rng: np.random.Generator,
) -> dict:
    B, sigma_sq = make_factor_model(p, r, rng)
    X = simulate_X(T, B, sigma_sq, rng)
    beta, beta_pre, beta_post, t_star = make_beta(p, s, tau, overlap, snr, T, rng)
    Y = simulate_Y(X, beta, rng)
    return {
        "X": X, "Y": Y, "beta": beta,
        "beta_pre": beta_pre, "beta_post": beta_post,
        "t_star": t_star, "B": B, "sigma_sq": sigma_sq,
    }
