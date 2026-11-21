"""Ideal state-vector QAOA (classical simulation; little-endian: bit j <-> qubit j)."""
import numpy as np
from scipy.optimize import minimize
from .qubo import all_values, qubo_to_ising, ising_scale


def normalised_energies(a, J):
    """QUBO energies divided by the closed-form Ising scale (hardware-computable)."""
    h, K, _ = qubo_to_ising(a, J)
    E = all_values(a, J)
    return (E - E.mean()) / ising_scale(h, K)


def x_mixer(psi, beta, n):
    """exp(-i beta H_M) with H_M = -sum_j X_j (|+>^n is its ground state), i.e. RX(-2 beta)."""
    c, s = np.cos(beta), 1j * np.sin(beta)
    for q in range(n):
        psi = psi.reshape(-1, 2, 1 << q)
        psi = c * psi + s * psi[:, ::-1, :]
    return psi.reshape(-1)


def qaoa_probs(En, gammas, betas):
    n = int(np.log2(len(En)))
    psi = np.full(len(En), 1 / np.sqrt(len(En)), complex)
    for g, b in zip(gammas, betas):
        psi = x_mixer(psi * np.exp(-1j * g * En), b, n)
    return np.abs(psi) ** 2


def ramp_angles(p, dg=0.6, db=0.5):
    """Linear-ramp (annealing-like) initial schedule."""
    t = (np.arange(p) + 0.5) / p
    return np.r_[dg * t, db * (1 - t)]


def interp_init(x):
    """INTERP heuristic: initial (p+1)-layer schedule from an optimised p-layer one."""
    p = len(x) // 2
    def up(v):
        t_old = (np.arange(p) + 0.5) / p; t_new = (np.arange(p + 1) + 0.5) / (p + 1)
        return np.interp(t_new, t_old, v) if p > 1 else np.repeat(v, 2)
    return np.r_[up(x[:p]), up(x[p:])]


def train_angles(En_list, p, seed=0, restarts=3, maxiter=400, x0=None):
    """Instance-averaged angle training: one schedule reused for all windows."""
    rng = np.random.default_rng(seed)

    def obj(x):
        return float(np.mean([qaoa_probs(E, x[:p], x[p:]) @ E for E in En_list]))
    best = None
    for r in range(restarts):
        start = (x0 if x0 is not None else ramp_angles(p, 2.5, 1.2))
        xi = start if r == 0 else start * rng.uniform(0.6, 1.4, 2 * p)
        res = minimize(obj, xi, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-5))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def sample_bits(probs, shots, rng):
    n = int(np.log2(len(probs)))
    idx = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    return ((idx[:, None] >> np.arange(n)) & 1).astype(int)


def cvar(values, alpha=0.1):
    v = np.sort(np.asarray(values)); k = max(1, int(np.ceil(alpha * len(v))))
    return float(v[:k].mean())
