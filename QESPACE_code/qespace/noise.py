"""Gate-level noisy simulation (Monte-Carlo Pauli trajectories) and error mitigation.

Noise model (deliberately simple, device-agnostic):
  * every two-qubit gate is followed, with probability p2, by a uniformly random
    non-identity two-qubit Pauli (two-qubit depolarising channel);
  * every physical one-qubit rotation is followed, with probability p1, by a
    random one-qubit Pauli; RZ is virtual and noiseless;
  * read-out flips each bit independently (p01: 0 read as 1, p10: 1 read as 0).

Mitigation implemented here mirrors what is switched on in Qiskit Runtime
(ibm/runtime.py): read-out mitigation (TREX-like rescaling), zero-noise
extrapolation by noise amplification, and sample-level post-selection.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class NoiseModel:
    p1: float = 2e-4
    p2: float = 2.5e-3          # of the order of Heron-class layered two-qubit error
    p01: float = 8e-3
    p10: float = 1.2e-2
    swap_overhead: float = 1.0  # two-qubit gates per logical interaction (routing)

    def scaled(self, lam):
        return NoiseModel(self.p1 * lam, self.p2 * lam, self.p01, self.p10, self.swap_overhead)

    @property
    def p2_eff(self):
        return 1.0 - (1.0 - self.p2) ** self.swap_overhead


class Sim:
    """State-vector simulator with Pauli-error injection. Little-endian."""

    def __init__(self, n, noise: NoiseModel = None, rng=None):
        self.n = n; self.noise = noise; self.rng = rng or np.random.default_rng()
        self.idx = np.arange(1 << n)
        self.zs = [1.0 - 2.0 * ((self.idx >> q) & 1) for q in range(n)]
        self.psi = np.zeros(1 << n, complex); self.psi[0] = 1.0
        self.n2q = 0

    # --- Paulis
    def _pauli(self, q, k):          # k: 1=X, 2=Y, 3=Z
        if k == 1:
            self.psi = self.psi[self.idx ^ (1 << q)]
        elif k == 3:
            self.psi = self.psi * self.zs[q]
        elif k == 2:
            self.psi = 1j * (self.psi * self.zs[q])[self.idx ^ (1 << q)]

    def _err1(self, q):
        if self.noise and self.rng.random() < self.noise.p1:
            self._pauli(q, int(self.rng.integers(1, 4)))

    def _err2(self, q1, q2):
        self.n2q += 1
        if self.noise and self.rng.random() < self.noise.p2_eff:
            k = int(self.rng.integers(1, 16))
            if k % 4: self._pauli(q1, k % 4)
            if k // 4: self._pauli(q2, k // 4)

    # --- gates (angle convention: exp(-i theta P))
    def h(self, q):
        m = 1 << q
        self.psi = (self.psi[self.idx & ~m] + self.zs[q] * self.psi[self.idx | m]) / np.sqrt(2)
        self._err1(q)

    def rz(self, q, th):
        self.psi = self.psi * (np.cos(th) - 1j * np.sin(th) * self.zs[q])

    def rx(self, q, th):
        self.psi = np.cos(th) * self.psi - 1j * np.sin(th) * self.psi[self.idx ^ (1 << q)]
        self._err1(q)

    def rzz(self, q1, q2, th):
        self.psi = self.psi * (np.cos(th) - 1j * np.sin(th) * self.zs[q1] * self.zs[q2])
        self._err2(q1, q2)

    def rxx(self, q1, q2, th):
        self.psi = np.cos(th) * self.psi - 1j * np.sin(th) * self.psi[self.idx ^ ((1 << q1) | (1 << q2))]
        self._err2(q1, q2)

    def cx(self, c, t):
        self.psi = self.psi[self.idx ^ (((self.idx >> c) & 1) << t)]
        self._err2(c, t)

    def probs(self):
        return np.abs(self.psi) ** 2


def qaoa_circuit(sim: Sim, h, K, gammas, betas, tol=1e-12):
    """Standard QAOA: exp(-i beta H_M) exp(-i gamma H_C), H_C = h.Z + sum K ZZ, H_M = -sum X."""
    n = sim.n
    for q in range(n):
        sim.h(q)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(K[i, j]) > tol]
    for g, b in zip(gammas, betas):
        for q in range(n):
            sim.rz(q, g * h[q])
        for (i, j) in pairs:
            sim.rzz(i, j, g * K[i, j])
        for q in range(n):
            sim.rx(q, -b)


def averaged_probs(build, n, noise, rng, trajectories=200):
    """Average outcome distribution over Pauli trajectories (= noisy channel output)."""
    acc = np.zeros(1 << n); n2q = 0
    if noise is None:
        s = Sim(n, None, rng); build(s); return s.probs(), s.n2q
    for _ in range(trajectories):
        s = Sim(n, noise, rng); build(s); acc += s.probs(); n2q = s.n2q
    return acc / trajectories, n2q


def sample_with_readout(probs, shots, noise, rng):
    n = int(np.log2(len(probs)))
    idx = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(int)
    if noise is not None:
        r = rng.random(bits.shape)
        flip = np.where(bits == 0, r < noise.p01, r < noise.p10)
        bits = bits ^ flip
    return bits


# ------------------------------------------------------------------ mitigation
def ising_expectation(bits, h, K, readout=None):
    """<H_C> from samples. With ``readout=(p01, p10)`` the one- and two-body Z
    correlators are corrected by inverting the single-qubit assignment matrices
    (tensored read-out mitigation; equals TREX rescaling when p01 = p10)."""
    z = 1.0 - 2.0 * bits
    m1 = z.mean(0)
    m2 = (z.T @ z) / len(z)
    if readout is not None:
        n = bits.shape[1]
        p01, p10 = (np.broadcast_to(np.asarray(v, float), (n,)) for v in readout)
        s = 1.0 - p01 - p10                 # <Z_i>_noisy = s_i <Z_i> + o_i
        o = p10 - p01
        t1 = (m1 - o) / s
        m2 = (m2 - np.outer(s * t1, o) - np.outer(o, s * t1) - np.outer(o, o)) / np.outer(s, s)
        m1 = t1
    return float(h @ m1 + np.sum(np.triu(K, 1) * m2))


def extrapolate(lams, vals, kind="linear"):
    """Zero-noise extrapolation of expectation values measured at noise factors lams."""
    lams = np.asarray(lams, float); vals = np.asarray(vals, float)
    if kind == "linear":
        return float(np.polyval(np.polyfit(lams, vals, 1), 0.0))
    if kind == "quadratic":
        return float(np.polyval(np.polyfit(lams, vals, min(2, len(lams) - 1)), 0.0))
    if kind == "exponential":
        # E(lam) = A exp(-b lam): valid when the fully-mixed value is ~0, as for a
        # traceless (centred) cost Hamiltonian. Falls back to linear if signs differ.
        if np.all(vals < 0) or np.all(vals > 0):
            sgn = np.sign(vals[0])
            c = np.polyfit(lams, np.log(np.abs(vals)), 1)
            return float(sgn * np.exp(c[1]))
        return extrapolate(lams, vals, "linear")
    raise ValueError(kind)
