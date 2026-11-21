"""QAOA protected by the [[k+2, k, 2]] "Iceberg" quantum error-detecting code.

Physical qubits: 0..k-1 data, k = "t" (top), k+1 = "b" (bottom); n = k+2 even.
Stabilisers:  S_X = X^{(x)n},  S_Z = Z^{(x)n}.
Logical operators:  Xbar_i = X_t X_i,   Zbar_i = Z_i Z_b.
Hence every QAOA gate is a native two-body rotation:
    exp(-i th Zbar_i)        = exp(-i th Z_i Z_b)
    exp(-i th Zbar_i Zbar_j) = exp(-i th Z_i Z_j)
    exp(-i th Xbar_i)        = exp(-i th X_t X_i)
Mixer convention: H_M = -sum Xbar_i (as in qaoa.py), hence the minus sign below.
Logical |+>^k is the X-basis GHZ state (|+..+> + |-..->)/sqrt(2).
Final Z read-out gives S_Z for free: odd parity => an error was detected and the
shot is discarded. Mid-circuit syndrome extraction of S_X/S_Z with two ancillas
is implemented for hardware in ibm/circuits.py.

This is error *detection* with post-selection, not fault-tolerant correction.
"""
import numpy as np
from .noise import Sim


def iceberg_qaoa(sim: Sim, h, K, gammas, betas, tol=1e-12):
    k = len(h); n = k + 2; t, b = k, k + 1
    assert sim.n == n and n % 2 == 0, "k must be even"
    # X-basis GHZ: Z-basis GHZ on a chain, then Hadamards
    sim.h(t)
    order = [t] + list(range(k)) + [b]
    for q1, q2 in zip(order[:-1], order[1:]):
        sim.cx(q1, q2)
    for q in range(n):
        sim.h(q)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k) if abs(K[i, j]) > tol]
    for g, be in zip(gammas, betas):
        for i in range(k):
            if abs(h[i]) > tol:
                sim.rzz(i, b, g * h[i])
        for (i, j) in pairs:
            sim.rzz(i, j, g * K[i, j])
        for i in range(k):
            sim.rxx(t, i, -be)


def decode(bits):
    """Physical samples [shots, k+2] -> (logical bits [accepted, k], accept mask)."""
    k = bits.shape[1] - 2
    ok = bits.sum(1) % 2 == 0
    logical = bits[:, :k] ^ bits[:, [k + 1]]
    return logical[ok], ok
