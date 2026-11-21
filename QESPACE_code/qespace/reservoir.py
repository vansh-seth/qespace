"""Collision-model quantum reservoir with physically realisable memory.

Each reservoir has n_m memory qubits (never reset) and n_a interface qubits.
Per time step t:
  1. reset interface qubits and encode the inputs:  RY(pi * s_t) on interface 0,
     RY(pi * m_t) on interface 1 (if a second input stream is present);
  2. apply a fixed unitary U = exp(-i H tau) on all qubits, H a random
     transverse-field Ising Hamiltonian;
  3. measure the interface qubits in Z (features <Z_a>, <Z_a Z_a'>), discard them.
Information about past inputs survives only in the memory qubits, so the map on
the interface is non-Markovian while every step is a legal quantum operation
(mid-circuit measurement + reset; no cloning of density matrices).

Exact density-matrix simulation; optional dephasing and finite-shot noise.
"""
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1.0, -1.0]).astype(complex)


def _op(single, q, n):
    out = np.array([[1.0]], complex)
    for k in range(n):
        out = np.kron(out, single if k == q else I2)
    return out


class CollisionReservoir:
    def __init__(self, n_m=3, n_a=2, tau=1.0, seed=0, dephase=0.0, trotter=False):
        self.n_m, self.n_a, self.n = n_m, n_a, n_m + n_a
        rng = np.random.default_rng(seed)
        n = self.n
        HX = np.zeros((1 << n, 1 << n), complex); HZ = np.zeros_like(HX)
        for i in range(n):
            HX += rng.uniform(0.5, 1.5) * _op(X, i, n)
            for j in range(i + 1, n):
                HZ += rng.uniform(-1, 1) * _op(Z, i, n) @ _op(Z, j, n)
        # trotter=True reproduces the hardware block of ibm/circuits.py (ZZ layer, then X layer)
        self.U = expm(-1j * tau * HX) @ expm(-1j * tau * HZ) if trotter else expm(-1j * tau * (HX + HZ))
        self.dephase = dephase
        # interface qubits are the *last* n_a tensor factors
        da = 1 << n_a
        za = [np.kron(np.eye(1 << n_m), _op(Z, q, n_a)) for q in range(n_a)]
        self.obs = list(za) + [za[i] @ za[j] for i in range(n_a) for j in range(i + 1, n_a)]
        self.zm = [_op(Z, q, n) for q in range(n_m)]

    @property
    def n_features(self):
        return len(self.obs)

    def run(self, inputs, shots=None, rng=None):
        """inputs: [T, d] in [0,1], d <= n_a. Returns features [T, n_features]."""
        inputs = np.atleast_2d(inputs.T).T
        dm = 1 << self.n_m
        rho_m = np.zeros((dm, dm), complex); rho_m[0, 0] = 1.0
        feats = np.zeros((len(inputs), self.n_features))
        for t, s in enumerate(inputs):
            anc = np.array([[1.0]], complex)
            for q in range(self.n_a):
                th = np.pi * (s[q] if q < len(s) else 0.0)
                v = np.array([np.cos(th / 2), np.sin(th / 2)], complex)
                anc = np.kron(anc, np.outer(v, v.conj()))
            rho = self.U @ np.kron(rho_m, anc) @ self.U.conj().T
            if self.dephase > 0:
                for zq in self.zm:
                    rho = (1 - self.dephase) * rho + self.dephase * zq @ rho @ zq
            feats[t] = [np.real(np.trace(rho @ o)) for o in self.obs]
            da = 1 << self.n_a
            rho_m = np.einsum("iaja->ij", rho.reshape(dm, da, dm, da))
        if shots:
            rng = rng or np.random.default_rng()
            feats = 2.0 * rng.binomial(shots, np.clip((1 + feats) / 2, 0, 1)) / shots - 1.0
        return feats


def quantum_features(inputs, n_res=4, n_m=3, n_a=2, taus=(0.6, 1.0, 1.4, 1.8), seed=0, **kw):
    """Spatial multiplexing: several small reservoirs with different interaction times."""
    return np.hstack([CollisionReservoir(n_m, n_a, taus[r % len(taus)], seed + r, kw.get("dephase", 0.0))
                      .run(inputs, kw.get("shots"), kw.get("rng")) for r in range(n_res)])


# ----------------------------------------------------------- classical baselines
def esn_features(inputs, n=12, rho=0.9, leak=0.5, in_scale=1.0, seed=0):
    inputs = np.atleast_2d(inputs.T).T
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, n)); W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.uniform(-1, 1, (n, inputs.shape[1])) * in_scale; b = rng.uniform(-0.5, 0.5, n)
    x = np.zeros(n); out = np.zeros((len(inputs), n))
    for t, s in enumerate(inputs):
        x = (1 - leak) * x + leak * np.tanh(W @ x + Win @ s + b); out[t] = x
    return out


def delay_features(inputs, taps=6):
    """Explicit tapped delay line (linear memory), same feature count as the others."""
    inputs = np.atleast_2d(inputs.T).T
    cols = []
    for d in range(inputs.shape[1]):
        for k in range(taps):
            cols.append(np.r_[np.zeros(k), inputs[:len(inputs) - k, d]])
    return np.column_stack(cols)


def ridge_eval(F, y, washout=50, n_train=1000, alpha=1e-6):
    """Train ridge read-out on [washout, n_train), report NMSE and r^2 on the rest."""
    Xd = np.column_stack([np.ones(len(F)), F])
    A, b = Xd[washout:n_train], y[washout:n_train]
    R = alpha * np.eye(Xd.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ b)
    pred = Xd[n_train:] @ w; yt = y[n_train:]
    nmse = float(np.mean((pred - yt) ** 2) / (np.var(yt) + 1e-15))
    r2 = float(np.corrcoef(pred, yt)[0, 1] ** 2) if np.std(pred) > 1e-12 else 0.0
    return nmse, r2


# ----------------------------------------------------------------------- tasks
def eligibility_target(s, m, lam=0.8, delay=1):
    """Three-factor credit signal: y_t = m_t * sum_k lam^k s_{t-delay-k}."""
    e = np.zeros(len(s)); acc = 0.0
    for t in range(len(s)):
        acc = lam * acc + (s[t - delay] if t >= delay else 0.0)
        e[t] = acc
    return m * e
