"""Budgeted plasticity allocation as a QUBO, and classical solvers.

f(u) = a.u + sum_{i<j} J_ij u_i u_j   subject to  c.u <= B,  u in {0,1}^n
"""
import numpy as np


def all_bits(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)


def value(u, a, J):
    u = np.asarray(u, float)
    return float(a @ u + u @ np.triu(J, 1) @ u)


def all_values(a, J):
    b = all_bits(len(a))
    return b @ a + np.einsum("bi,ij,bj->b", b, np.triu(J, 1), b)


def exact_feasible(a, J, cost, B):
    n = len(a)
    if n > 22:
        raise ValueError("enumeration capped at n<=22")
    b = all_bits(n)
    v = b @ a + np.einsum("bi,ij,bj->b", b, np.triu(J, 1), b)
    v = np.where(b @ cost <= B + 1e-9, v, np.inf)
    k = int(np.argmin(v))
    return b[k].astype(int), float(v[k])


def knapsack_diag(a, J, cost, B):
    """Exact optimum of the *diagonal-only* model (ignores J): 0/1 knapsack DP."""
    n, B = len(a), int(B)
    best = np.zeros(B + 1); choice = [[() for _ in range(B + 1)]]
    sel = [frozenset() for _ in range(B + 1)]
    for i in range(n):
        if a[i] >= 0:
            continue
        c = int(cost[i])
        for w in range(B, c - 1, -1):
            if best[w - c] + a[i] < best[w] - 1e-15:
                best[w] = best[w - c] + a[i]; sel[w] = sel[w - c] | {i}
    u = np.zeros(n, int); u[list(sel[B])] = 1
    return u, value(u, a, J)


def greedy_feasible(a, J, cost, B):
    n = len(a); u = np.zeros(n, int); rem = B; cur = 0.0
    Js = np.triu(J, 1); Js = Js + Js.T
    while True:
        gain = -(a + Js @ u)                      # marginal decrease if switched on
        ok = (u == 0) & (cost <= rem) & (gain > 1e-12)
        if not ok.any():
            break
        i = int(np.argmax(np.where(ok, gain / np.maximum(cost, 1), -np.inf)))
        u[i] = 1; rem -= int(cost[i])
    return u, value(u, a, J)


def simulated_annealing(a, J, cost, B, rng, iters=3000, T0=None, T1=None):
    n = len(a); Js = np.triu(J, 1); Js = Js + Js.T
    scale = np.abs(a).mean() + 1e-12
    T0 = T0 or 2.0 * scale; T1 = T1 or 0.01 * scale
    u = np.zeros(n); cur = 0.0; csum = 0.0; best, best_v = u.copy(), 0.0
    for s in range(iters):
        T = T0 * (T1 / T0) ** (s / (iters - 1))
        k = rng.integers(n)
        d = (1 - 2 * u[k]) * (a[k] + Js[k] @ u)
        dc = (1 - 2 * u[k]) * cost[k]
        if csum + dc > B:
            continue
        if d <= 0 or rng.random() < np.exp(-d / T):
            u[k] = 1 - u[k]; cur += d; csum += dc
            if cur < best_v - 1e-15:
                best, best_v = u.copy(), cur
    return best.astype(int), value(best, a, J)


def random_feasible(a, J, cost, B, rng):
    """One random feasible mask: random order, add while budget allows."""
    n = len(a); u = np.zeros(n, int); rem = B
    for i in rng.permutation(n):
        if cost[i] <= rem and rng.random() < 0.5:
            u[i] = 1; rem -= cost[i]
    return u, value(u, a, J)


def best_of_samples(bitrows, a, J, cost, B):
    """Classical post-processing shared by every sampler (quantum or not):
    discard infeasible samples, return best feasible; all-zero is the fallback."""
    bitrows = np.asarray(bitrows, float)[:, :len(a)]
    v = bitrows @ a + np.einsum("bi,ij,bj->b", bitrows, np.triu(J, 1), bitrows)
    feas = bitrows @ cost <= B + 1e-9
    if not feas.any():
        return np.zeros(len(a), int), 0.0, 0.0
    v = np.where(feas, v, np.inf); k = int(np.argmin(v))
    if v[k] > 0:
        return np.zeros(len(a), int), 0.0, float(feas.mean())
    return bitrows[k].astype(int), float(v[k]), float(feas.mean())


# ------------------------------------------------------------ constraint encodings
def slack_qubo(a, J, cost, B, P):
    """Binary slack: F = f + P (c.u + sum 2^h b_h - B)^2 ; n + ceil(log2(B+1)) bits."""
    n = len(a); k = max(1, int(B).bit_length())
    d = np.r_[np.asarray(cost, float), 2.0 ** np.arange(k)]
    a2 = np.r_[a, np.zeros(k)] + P * (d ** 2 - 2.0 * B * d)
    J2 = np.zeros((n + k, n + k)); J2[:n, :n] = np.triu(J, 1)
    J2 += 2.0 * P * np.triu(np.outer(d, d), 1)
    return a2, J2, P * B ** 2


def unbalanced_qubo(a, J, cost, B, lam1, lam2):
    """Unbalanced penalisation (no slack qubits): F = f - lam1 h + lam2 h^2, h = B - c.u."""
    c = np.asarray(cost, float)
    a2 = a + lam1 * c + lam2 * (c ** 2 - 2.0 * B * c)
    J2 = np.triu(J, 1) + 2.0 * lam2 * np.triu(np.outer(c, c), 1)
    return a2, J2, -lam1 * B + lam2 * B ** 2


def qubo_to_ising(a, J):
    """z_i = (1 - Z_i)/2  ->  H_C = sum h_i Z_i + sum_{i<j} K_ij Z_i Z_j + const."""
    Ju = np.triu(J, 1); Js = Ju + Ju.T
    h = -a / 2.0 - Js.sum(1) / 4.0
    K = Ju / 4.0
    const = a.sum() / 2.0 + Ju.sum() / 4.0
    return h, K, const


def ising_scale(h, K):
    """Standard deviation of H_C over uniformly random bitstrings (closed form)."""
    return float(np.sqrt(np.sum(h ** 2) + np.sum(np.triu(K, 1) ** 2))) + 1e-15


# ------------------------------------------------------------ hardware-aware helpers
def swap_network_schedule(n):
    """Odd-even transposition network on a line: n brick layers; every pair of
    decision variables is adjacent exactly once. Returns (layers, final_perm) with
    layers[l] = [(position, var_u, var_v), ...] and perm[position] = variable."""
    perm = list(range(n)); layers = []
    for layer in range(n):
        row = []
        for pos in range(layer % 2, n - 1, 2):
            row.append((pos, perm[pos], perm[pos + 1]))
            perm[pos], perm[pos + 1] = perm[pos + 1], perm[pos]
        layers.append(row)
    return layers, perm


def truncate_to_grid(J, rows, cols):
    """Keep only couplings between nearest neighbours of a rows x cols lattice
    (variable k sits at row k // cols, column k % cols): Nighthawk-native problem."""
    n = rows * cols; Jt = np.zeros_like(J)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i // cols - j // cols) + abs(i % cols - j % cols) == 1:
                Jt[i, j] = J[i, j]
    return Jt


def gate_counts(n, p, mode, rows=None, cols=None):
    """Analytic two-qubit resources per circuit (before device-specific optimisation).
    mode 'swap': dense problem, line SWAP network, RZZ*SWAP = 3 CZ.
    mode 'grid': lattice-native problem, one RZZ = 2 CZ (1 with fractional gates)."""
    if mode == "swap":
        return dict(two_qubit_gates=3 * n * (n - 1) // 2 * p, two_qubit_depth=3 * n * p)
    e = 2 * rows * cols - rows - cols
    return dict(two_qubit_gates=2 * e * p, two_qubit_depth=2 * 4 * p)
