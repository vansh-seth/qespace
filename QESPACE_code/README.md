# Q-ESPACE research code

Quantum-assisted Energy-aware Stochastic Plasticity and Consolidation Engine.
Code for the manuscript draft "Quantum-assisted allocation of metabolically constrained
synaptic plasticity for continual learning in spiking networks".

## What is here

| Path | Purpose | Needs |
|---|---|---|
| `qespace/snn.py` | Recurrent LIF network, stochastic vesicle release, eligibility traces, branch-level QUBO coefficients | NumPy |
| `qespace/qubo.py` | Allocation problem, exact / knapsack / greedy / annealing solvers, slack and unbalanced encodings, Ising map, SWAP-network schedule, lattice truncation, gate counts | NumPy |
| `qespace/qaoa.py` | State-vector QAOA, layer-wise angle training, sampling, CVaR | NumPy, SciPy |
| `qespace/noise.py` | Gate-level noisy simulator (Pauli trajectories), read-out mitigation, zero-noise extrapolation | NumPy |
| `qespace/iceberg.py` | QAOA in the [[k+2,k,2]] error-detecting code | NumPy |
| `qespace/reservoir.py` | Collision-model quantum reservoir, echo-state and delay-line baselines | NumPy, SciPy |
| `qespace/drawing.py` | Circuit drawings for the figures | Matplotlib |
| `experiments/exp1_continual.py` | Continual learning under a plasticity budget (Fig. 3) | |
| `experiments/exp2_qubo.py` | Sampler benchmark, n = 12, 16, 20 (Fig. 4) | about 30 min, 2 GB RAM |
| `experiments/exp3_noise.py` | Noise, mitigation, error detection (Fig. 5) | |
| `experiments/exp4_reservoir.py` | Reservoir against classical memory (Fig. 6) | |
| `experiments/exp5_resources.py` | Gate counts and lattice truncation (Table 2) | |
| `experiments/make_figures.py` | All figures from `results/*.json` | Matplotlib |
| `ibm/circuits.py` | Qiskit circuits: QAOA, SWAP network, error-detected QAOA, reservoir, calibration | Qiskit 2.x |
| `ibm/runtime.py` | Backend selection, transpilation, DD, twirling, TREX, ZNE | qiskit-ibm-runtime |
| `ibm/run_hardware.py` | Command-line runner for IBM Nighthawk / Heron or a local fake backend | qiskit-ibm-runtime, qiskit-aer |
| `tests/test_all.py` | 10 unit and consistency tests | pytest |
| `results/` | JSON output of every experiment reported in the draft, plus the extracted instances | |

## Reproduce the simulations

```bash
python -m pip install -r requirements.txt
python -m pytest -q                              # 10 tests
python -m experiments.exp1_continual --seeds 20  # about 5 min
python -m experiments.exp2_qubo                  # about 30 min
python -m experiments.exp3_noise
python -m experiments.exp4_reservoir --seeds 10
python -m experiments.exp5_resources
python -m experiments.make_figures
```

Run them in this order: exp3 reads the angles written by exp1, exp5 reads the instances written by exp2.

## Run on IBM Quantum

```bash
python -m pip install -r requirements-ibm.txt
# 1. dry run on a local noisy Heron snapshot, no account needed
python -m ibm.run_hardware --fake --task qaoa --instances results/instances_n12.json --max 5
# 2. save credentials once (Python):
#    from qiskit_ibm_runtime import QiskitRuntimeService
#    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...", instance="...")
# 3. hardware
python -m ibm.run_hardware --backend ibm_miami  --task qaoa      --instances results/instances_n12.json
python -m ibm.run_hardware --backend ibm_boston --task estimator --instances results/instances_n12.json
python -m ibm.run_hardware --backend ibm_boston --task iceberg   --instances results/instances_n10.json
python -m ibm.run_hardware --backend ibm_boston --task reservoir
```

Each run writes `results/hardware/<task>_<backend>_<time>.json` with raw metrics, transpiled
gate counts, physical qubits, job ids and timing.

## Status, stated plainly

* Everything in `qespace/` and `experiments/` was run; the numbers in the draft come from `results/`.
* **Nothing in `ibm/` has been executed.** Qiskit was not installed where this package was
  written. The files compile as Python, and the circuit conventions are tested against the
  NumPy simulators, but API details (option names, backend names, result accessors) must be
  checked with the `--fake` dry run before any hardware time is spent.
* "Quantum" results in `results/` are classical simulations. Noise is a depolarizing model,
  not device data.
* Energies are dimensionless proxy units, not joules or ATP counts.
* Conventions: bit j is qubit j (little-endian); `H_M = -sum X`, so the mixer gate is `RX(-2 beta)`.
