"""Qiskit Runtime helpers: backend choice, transpilation, error suppression/mitigation.

Requires:  pip install "qiskit>=2.0" "qiskit-ibm-runtime>=0.40" qiskit-aer
Credentials: QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
             token="<API key>", instance="<CRN>")  once on your machine.

Option names follow the qiskit-ibm-runtime V2 primitives documentation
(SamplerOptions / EstimatorOptions). They change between releases: if an option
is rejected, check https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/options
"""
import time
import numpy as np

try:
    from qiskit.transpiler import generate_preset_pass_manager
except ImportError:                                   # older Qiskit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Backends named in the manuscript (availability depends on your IBM plan).
NIGHTHAWK = "ibm_miami"       # 120 qubits, square lattice, 218 couplers
HERON_R3 = "ibm_boston"       # 156 qubits, heavy-hex, 176 couplers; dynamic circuits


def get_backend(name=None, fake=False, fractional_gates=False):
    """fake=True -> local noisy simulator built from a Heron snapshot (no account needed)."""
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeFez      # Heron r2 snapshot
        from qiskit_aer import AerSimulator
        return AerSimulator.from_backend(FakeFez())
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    if name is None:
        return service.least_busy(operational=True, simulator=False, min_num_qubits=100)
    # fractional gates give native RZZ(theta) on Heron; check the feature-compatibility
    # table before combining them with twirling / dynamic circuits.
    return service.backend(name, use_fractional_gates=fractional_gates)


def to_isa(circuits, backend, optimization_level=3, seed=11):
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      seed_transpiler=seed)
    out = pm.run(circuits)
    return out


def circuit_report(isa):
    two = isa.depth(lambda inst: inst.operation.num_qubits == 2)
    ops = dict(isa.count_ops())
    n2 = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz", "swap"))
    return dict(depth=isa.depth(), two_qubit_depth=two, two_qubit_gates=n2, ops=ops,
                physical_qubits=sorted({isa.find_bit(q).index for inst in isa.data for q in inst.qubits}))


def make_sampler(backend, shots=4096, dd=True, twirl=True):
    """Sampler with error *suppression*: dynamical decoupling + Pauli twirling.
    (Samplers return bitstrings; read-out mitigation is applied in post-processing.)"""
    from qiskit_ibm_runtime import SamplerV2
    s = SamplerV2(mode=backend)
    s.options.default_shots = shots
    try:
        if dd:
            s.options.dynamical_decoupling.enable = True
            s.options.dynamical_decoupling.sequence_type = "XpXm"
        if twirl:
            s.options.twirling.enable_gates = True
            s.options.twirling.enable_measure = True
            s.options.twirling.num_randomizations = "auto"
    except Exception as e:                           # local simulators ignore these options
        print("note: suppression options not applied:", e)
    return s


def make_estimator(backend, zne=True, noise_factors=(1, 3, 5), extrapolator=("exponential", "linear"),
                   precision=0.01):
    """Estimator with TREX read-out mitigation, gate twirling, DD and ZNE."""
    from qiskit_ibm_runtime import EstimatorV2
    e = EstimatorV2(mode=backend)
    e.options.default_precision = precision
    try:
        e.options.dynamical_decoupling.enable = True
        e.options.dynamical_decoupling.sequence_type = "XpXm"
        e.options.twirling.enable_gates = True
        e.options.resilience.measure_mitigation = True          # TREX
        if zne:
            e.options.resilience.zne_mitigation = True
            e.options.resilience.zne.noise_factors = tuple(noise_factors)
            e.options.resilience.zne.extrapolator = extrapolator
            # e.options.resilience.zne.amplifier = "pea"         # probabilistic error amplification
    except Exception as ex:
        print("note: resilience options not applied:", ex)
    return e


def counts_to_bits(counts, n):
    """Qiskit bitstrings are little-endian: rightmost character = classical bit 0."""
    rows = []
    for s, c in counts.items():
        s = s.replace(" ", "")
        rows += [[int(ch) for ch in s[::-1][:n]]] * int(c)
    return np.array(rows, dtype=int)


def job_metadata(job, backend):
    md = dict(job_id=job.job_id(), backend=getattr(backend, "name", str(backend)),
              retrieved_unix=time.time())
    for attr in ("metrics", "usage"):
        try:
            md[attr] = getattr(job, attr)()
        except Exception:
            pass
    return md


def readout_errors(counts0, counts1, n):
    """Per-qubit p01 (0 read as 1) and p10 (1 read as 0) from the calibration pair."""
    b0, b1 = counts_to_bits(counts0, n), counts_to_bits(counts1, n)
    return b0.mean(0), 1.0 - b1.mean(0)
