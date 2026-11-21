"""Qiskit circuit builders for Q-ESPACE (Qiskit >= 1.3 / 2.x).

NOT executed in the environment where this package was generated (Qiskit was
not installed there). Validate locally first:  python -m ibm.run_hardware --fake

Conventions (identical to qespace/qaoa.py and qespace/noise.py)
  H_C = sum_i h_i Z_i + sum_{i<j} K_ij Z_i Z_j         (bit z_i = (1 - Z_i)/2)
  H_M = - sum_i X_i
  layer = exp(-i beta H_M) exp(-i gamma H_C)
        = RX(-2 beta) on every qubit  *  RZZ(2 gamma K_ij)  *  RZ(2 gamma h_i)
  Qiskit: RZ(t) = exp(-i t Z/2), RZZ(t) = exp(-i t ZZ/2), RX(t) = exp(-i t X/2).
Classical bit j always holds decision variable j.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

TOL = 1e-12


def _pairs(K):
    n = len(K)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if abs(K[i, j]) > TOL]


def qaoa_circuit(h, K, p, measure=True):
    """Plain QAOA; the transpiler routes the ZZ interactions.

    Use for lattice-native (sparse) problems, and for EstimatorV2 runs."""
    n = len(h)
    g, b = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for l in range(p):
        for i in range(n):
            if abs(h[i]) > TOL:
                qc.rz(2.0 * float(h[i]) * g[l], i)
        for (i, j) in _pairs(K):
            qc.rzz(2.0 * float(K[i, j]) * g[l], i, j)
        for i in range(n):
            qc.rx(-2.0 * b[l], i)
        qc.barrier()
    if measure:
        qc.measure_all()                      # register "meas", clbit j <- qubit j
    return qc


def swap_network_qaoa(h, K, p):
    """Dense problems on a line of qubits: odd-even SWAP network.

    n brick layers of (RZZ + SWAP) on neighbours let every pair interact once
    per QAOA layer with nearest-neighbour gates only. Works on heavy-hex (Heron)
    and square (Nighthawk) lattices, which both contain long lines.
    Two-qubit depth per QAOA layer: n blocks, each RZZ*SWAP = 3 CZ after synthesis."""
    n = len(h)
    g, b = ParameterVector("gamma", p), ParameterVector("beta", p)
    qr, cr = QuantumRegister(n, "q"), ClassicalRegister(n, "meas")
    qc = QuantumCircuit(qr, cr)
    from qespace.qubo import swap_network_schedule
    perm = list(range(n))                     # perm[position] = decision variable
    qc.h(qr)
    for l in range(p):
        for pos in range(n):
            if abs(h[perm[pos]]) > TOL:
                qc.rz(2.0 * float(h[perm[pos]]) * g[l], pos)
        layers, _ = swap_network_schedule(n)
        for row in layers:
            for (pos, u, v) in row:
                assert (perm[pos], perm[pos + 1]) == (u, v) or l > 0
                u, v = perm[pos], perm[pos + 1]
                i, j = min(u, v), max(u, v)
                if abs(K[i, j]) > TOL:
                    qc.rzz(2.0 * float(K[i, j]) * g[l], pos, pos + 1)
                qc.swap(pos, pos + 1)
                perm[pos], perm[pos + 1] = v, u
        for pos in range(n):
            qc.rx(-2.0 * b[l], pos)
        qc.barrier()
    for pos in range(n):
        qc.measure(pos, perm[pos])            # undo the permutation classically
    return qc


def cost_observable(h, K):
    """H_C as a SparsePauliOp (for EstimatorV2 with ZNE / TREX)."""
    n = len(h)
    terms = [("Z", [i], float(h[i])) for i in range(n) if abs(h[i]) > TOL]
    terms += [("ZZ", [i, j], float(K[i, j])) for (i, j) in _pairs(K)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def grid_edges(rows, cols):
    """Nearest-neighbour edges of a rows x cols square lattice (Nighthawk-native)."""
    idx = lambda r, c: r * cols + c
    e = [(idx(r, c), idx(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    e += [(idx(r, c), idx(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    return e


# ----------------------------------------------------------- Iceberg [[k+2,k,2]]
def iceberg_qaoa(h, K, p, syndrome_rounds=0):
    """Error-detected QAOA. Qubits 0..k-1 data, k = top (t), k+1 = bottom (b),
    k+2 / k+3 = syndrome ancillas (only if syndrome_rounds > 0).

    Logical gates:  exp(-i a Zbar_i) = exp(-i a Z_i Z_b),
                    exp(-i a Zbar_i Zbar_j) = exp(-i a Z_i Z_j),
                    exp(-i a Xbar_i) = exp(-i a X_t X_i).
    Accept a shot iff the final Z parity of the k+2 code qubits is even and all
    mid-circuit syndromes are 0. Decode with  z_i = meas_i XOR meas_b.

    The syndrome gadget here is the simple (non-flagged) one. For publication
    replace it by the flag-fault-tolerant gadget of the Iceberg-code paper.
    The top/bottom qubits talk to every data qubit, so on heavy-hex or square
    lattices routing overhead is substantial: check the transpiled depth."""
    k = len(h); assert k % 2 == 0, "k must be even"
    t, bq = k, k + 1
    g, b = ParameterVector("gamma", p), ParameterVector("beta", p)
    code = QuantumRegister(k + 2, "q")
    regs = [code]
    if syndrome_rounds:
        anc = QuantumRegister(2, "anc"); regs.append(anc)
        syn = ClassicalRegister(2 * syndrome_rounds, "syn"); regs.append(syn)
    meas = ClassicalRegister(k + 2, "meas"); regs.append(meas)
    qc = QuantumCircuit(*regs)
    # logical |+>^k  =  X-basis GHZ on all k+2 qubits
    order = [t] + list(range(k)) + [bq]
    qc.h(t)
    for q1, q2 in zip(order[:-1], order[1:]):
        qc.cx(q1, q2)
    qc.h(code)
    for l in range(p):
        for i in range(k):
            if abs(h[i]) > TOL:
                qc.rzz(2.0 * float(h[i]) * g[l], i, bq)
        for (i, j) in _pairs(K):
            qc.rzz(2.0 * float(K[i, j]) * g[l], i, j)
        for i in range(k):
            qc.rxx(-2.0 * b[l], t, i)
        if syndrome_rounds and l < syndrome_rounds:
            az, ax = anc[0], anc[1]
            qc.reset(az); qc.reset(ax)
            for q in range(k + 2):            # S_Z = Z^(x)n
                qc.cx(q, az)
            qc.h(ax)
            for q in range(k + 2):            # S_X = X^(x)n
                qc.cx(ax, q)
            qc.h(ax)
            qc.measure(az, syn[2 * l]); qc.measure(ax, syn[2 * l + 1])
        qc.barrier()
    qc.measure(code, meas)
    return qc


# ------------------------------------------------- collision-model reservoir
def reservoir_circuit(inputs, n_m=3, n_a=2, tau=1.0, seed=0, trotter_steps=1):
    """One circuit for a whole input sequence (mid-circuit measurement + reset).

    inputs: array [T, d] with values in [0, 1], d <= n_a.
    Qubits 0..n_m-1 = memory (never reset), n_m..n_m+n_a-1 = interface.
    Classical register "feat" has T*n_a bits: bit t*n_a + q = interface q at step t.
    The random Hamiltonian is drawn exactly as in qespace.reservoir.CollisionReservoir
    (same seed -> same couplings); the block is first-order Trotterised, matching
    CollisionReservoir(..., trotter=True).

    Needs mid-circuit measurement and reset: use a Heron-class backend
    (ibm_miami / Nighthawk did not support dynamic circuits at release)."""
    inputs = np.atleast_2d(np.asarray(inputs, float).T).T
    T = len(inputs); n = n_m + n_a
    rng = np.random.default_rng(seed)
    ax, Jzz = np.zeros(n), np.zeros((n, n))
    for i in range(n):
        ax[i] = rng.uniform(0.5, 1.5)
        for j in range(i + 1, n):
            Jzz[i, j] = rng.uniform(-1, 1)
    qr, cr = QuantumRegister(n, "q"), ClassicalRegister(T * n_a, "feat")
    qc = QuantumCircuit(qr, cr)
    dt = tau / trotter_steps
    for t in range(T):
        for q in range(n_a):
            if t > 0:
                qc.reset(n_m + q)
            s = inputs[t, q] if q < inputs.shape[1] else 0.0
            qc.ry(np.pi * float(s), n_m + q)
        for _ in range(trotter_steps):
            for i in range(n):
                for j in range(i + 1, n):
                    qc.rzz(2.0 * Jzz[i, j] * dt, i, j)
            for i in range(n):
                qc.rx(2.0 * ax[i] * dt, i)
        for q in range(n_a):
            qc.measure(n_m + q, cr[t * n_a + q])
    return qc


def readout_calibration(n):
    """Two circuits (all-0, all-1) to estimate per-qubit assignment errors."""
    c0 = QuantumCircuit(n); c0.measure_all()
    c1 = QuantumCircuit(n); c1.x(range(n)); c1.measure_all()
    return [c0, c1]
