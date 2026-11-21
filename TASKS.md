# Q-ESPACE: computational work needed to complete the paper

Draft reviewed: *Quantum-assisted allocation of metabolically constrained synaptic plasticity
for continual learning in spiking networks* (manuscript, Overleaf source, code package and
completion guide, version of 14 Nov 2026).

Each task has an ID, where it runs, what it depends on, and which part of the manuscript it
fills. **Where** is one of:

* **Hellbender**: classical simulation or HPC job (Slurm). No quantum hardware.
* **QPU**: a job on a real quantum processor.
* **Laptop**: light analysis or writing that needs neither.

Priority: **P0** blocks submission of the present draft; **P1** is needed for a competitive
Nature Communications submission (completion guide, Section 4); **P2** is optional.

---

## 0. Decisions needed before work starts

### D1: Confirm which quantum hardware we can actually use (P0)
**Owner:** Varshith + PI. **Blocks:** every QPU task.
**Status (21 Nov 2026):** a collaborator has agreed to help with QPU account/access and
with running the Q-tasks below on real hardware. Once their access details are known, fill
in the bullets below and close this task before any Q-task is scheduled.

* The draft names **ibm_miami** (Nighthawk, 120 qubits) and **ibm_boston** (Heron r3).
  As of the January 2026 announcement, Nighthawk (ibm_miami) was early access for Premium
  and Flex plans only. A newer Nighthawk r2 (ibm_phoenix) is also listed for paid plans.
  The free Open Plan gives about 10 QPU minutes per 28 days, on Heron devices
  (e.g. ibm_kingston), with a one-time 180-minute offer for eligible users.
* Record: provider, plan / instance (CRN), devices visible to the account, monthly QPU
  minutes, whether dynamic circuits (mid-circuit measurement + reset) are enabled on each.
* The MQT SC26 tutorial
  (`munich-quantum-toolkit/.github`, branch `tutorial-SC26/tutorial`) teaches MQT Core
  compilation and job submission through **QDMI**; its notebooks run on local simulators and
  do not by themselves provide hardware access. If our hardware route is a QDMI device
  (IBM-QDMI, IQM-QDMI or Braket-QDMI), the runner in `ibm/` (Qiskit Runtime primitives)
  must be ported: see task **Q0**.
* If only Heron is available, the paper's Nighthawk claims (square-lattice, 120-qubit
  lattice-native run) must be rewritten around heavy-hex, or dropped.

### D2: Hellbender allocation (P0)
**Owner:** Varshith. Account and project, partitions (CPU, high-memory, GPU), wall-time
limits, scratch quota, Python 3.11+ environment, and whether a **Gurobi** academic licence
is available on the cluster (needed for A6). Write a Slurm batch script for A1 once the
account/partition are known (template: `#SBATCH --account=... --partition=... --cpus-per-task=8
--mem=16G --time=04:00:00`, then the commands under A1).

### D3: Scope (P0)
**Owner:** PI. Two routes, which decide how many P1 tasks are needed:

1. **Finish the present draft**: P0 tasks only (reproduce, validate, hardware at 10–12
   branches, fill placeholders). Realistic targets: npj Quantum Information, Communications
   Physics, Quantum Sci. Technol., Neuromorphic Computing and Engineering.
2. **Nature Communications**: add the 100–120-branch scaling experiment against tuned
   classical solvers (A5–A8, Q7) and ideally the recurrent-plasticity and benchmark
   extensions (A9, A10). The completion guide calls the scaling run "the experiment that
   decides the paper".

---

## A. Simulations on Hellbender

### A1: Reproduce all reported simulations (P0)
**Where:** Hellbender (CPU, 1 node, ~2 GB RAM). **Effort:** half a day. **Fills:** every
number in Results, Figs 3–6, Tables 1–2.

```bash
pip install -r requirements.txt
python -m pytest -q                              # 10 tests (passed in our check, 21 Nov)
python -m experiments.exp1_continual --seeds 20  # Fig. 3, ~5 min
python -m experiments.exp2_qubo                  # Fig. 4, ~30 min, 2 GB
python -m experiments.exp3_noise                 # Fig. 5
python -m experiments.exp4_reservoir --seeds 10  # Fig. 6
python -m experiments.exp5_resources             # Table 2
python -m experiments.make_figures
```
Order matters (exp3 reads angles from exp1; exp5 reads instances from exp2). Diff every
output against `results/*.json`; list any number that changes (NumPy version differences
are possible) so the text can be updated. Record package versions.

### A2: Validate the IBM code on local noisy fake backends (P0)
**Where:** Hellbender (CPU; the local noisy simulation is slow, ~40 s per 12-qubit circuit
at 256 shots on 2 cores). **Depends:** A1. **Blocks:** all QPU tasks.

Dry run of every task, all with `--fake`:
`qaoa` (swap and `--routing auto`, p = 1, 2), `estimator`, `iceberg`, `reservoir`.

Issues already found in a first dry run (Qiskit 2.6.2, qiskit-ibm-runtime 0.52.1,
qiskit-aer 0.18.4):

| # | Finding | Fix |
|---|---|---|
| a | `--fake` uses `FakeFez` (Heron r2), not the processors named in the paper. `FakeMiami`, `FakeNighthawk` and `FakeBoston` exist in qiskit-ibm-runtime 0.52.1. | Add `--fake-backend` option; dry-run against FakeMiami and FakeBoston. |
| b | `--task qaoa` (SWAP network) runs. Transpiled 12-branch, p = 2 circuit on FakeFez has **444** two-qubit gates (two-qubit depth 129), against **396** in Table 2. | Find where the extra 48 gates come from (line embedding on heavy-hex, decomposition of the SWAP+RZZ blocks); report transpiled counts per backend in Table 2. |
| c | `--task estimator` **fails**: `AerError: a circuit requires more memory than max_memory_mb` (the full-width transpiled circuit is simulated). | Use Aer `method="matrix_product_state"` for the fake run, or strip idle qubits before simulation; run on a high-memory node. |
| d | `--task iceberg` runs. Accepted fraction 0.45 on FakeFez (1 instance, 128 shots), far below the 0.77 simulated without routing in the draft. | Expected: routing overhead. Quantify over all 10 instances; this changes the error-detection paragraph. |
| f | `--task reservoir` runs (mid-circuit measurement + reset). Feature RMSE vs ideal = 0.144 on FakeFez at 128 shots. | Repeat at 2,048+ shots on FakeBoston; this sets the expectation for Q6. |
| g | `requirements.txt` does not include Qiskit; `requirements-ibm.txt` is needed for anything in `ibm/`. | Document. |

Also check by hand: Qiskit bit ordering in `counts_to_bits`, parameter binding by name,
SamplerV2/EstimatorV2 option names (`dynamical_decoupling`, `twirling`, `resilience.zne`)
against the installed runtime version, and that the fake-backend regret for one instance is
close to the simulated value in `results/exp3_noise.json`.

### A3: Pre-hardware predictions with device noise models (P0)
**Where:** Hellbender (CPU, high-memory). **Depends:** A2. **Fills:** the "simulation" curves
plotted next to hardware points in Figs 4, 5, 6.

For every QPU task (Q2–Q6), simulate the exact transpiled circuits with the noise model of
the target device (FakeMiami / FakeBoston, or a noise model built from the live calibration
on the day of the run). Output: expected p_opt, feasible fraction, best-of-256 regret,
⟨H_C⟩ bias, accepted fraction, reservoir feature RMSE. These numbers also fix the shot
counts and let us stop a hardware run early if it is clearly broken.

### A4: Re-run the noise study with device-derived noise (P2)
**Where:** Hellbender. Re-run `exp3_noise` with noise models built from Heron / Nighthawk
calibrations (including measured readout asymmetry and, where available, crosstalk) instead
of uniform depolarizing noise. Addresses a limitation stated in the Discussion.

### A5: Generate 100–120-branch lattice-native instances (P1)
**Where:** Hellbender (CPU, Slurm array over seeds). **Blocks:** A6, A7, A8, Q7.

* New code: networks with `n_groups` = 100 and 120 (`n_hid` = 8 × n), branches placed on a
  10 × 10 and 10 × 12 grid; keep only nearest-neighbour couplings (`qespace.qubo` already
  has lattice truncation).
* Design decision to record: how the plasticity budget (`A_max`, `atp_refill`, currently 12
  and 6) scales with n.
* Map the grid onto the actual Nighthawk coupling map from D1 (check for missing/bad
  couplers in the live calibration).
* Save 20–40 instances per size in the same JSON format as `results/instances_n*.json`.

### A6: Tuned classical baselines at 100–120 branches (P1)
**Where:** Hellbender (CPU, Slurm array; Gurobi licence from D2). **Depends:** A5.

On the same instances: Gurobi or OR-Tools CP-SAT (optimum or proven bound, time to
optimum), parallel tempering, tabu search, tuned simulated annealing, greedy. Record
time-to-target (99 % confidence) and regret at fixed time budgets. Without an optimum, the
regret of the QPU samples cannot be reported.

### A7: Tensor-network simulation of the 120-qubit QAOA circuits (P1)
**Where:** Hellbender (CPU high-memory, or GPU if a GPU tensor-network library is used).
**Depends:** A5.

MPS / tensor-network simulation of the lattice-native circuits at p = 1–3 (bond-dimension
convergence study). Gives (i) the noiseless reference for Q7, (ii) a test of whether the
circuit is classically simulable at this size, which reviewers will ask about.

### A8: Angles for the large instances (P1)
**Where:** Hellbender. **Depends:** A5, A7.

Check whether the fixed angles trained on 12-branch instances transfer to 100–120 branches
(the draft shows transfer up to n = 20). If not, train p = 1 angles analytically / by
light-cone simulation and p = 2–3 with A7. Angles must be fixed before Q7.

### A9: Plasticity in the recurrent layer (P1)
**Where:** Hellbender (GPU recommended). Extend learning from output synapses to recurrent
synapses (e-prop or surrogate gradients) and repeat the J = 0 ablation. Tests whether the
couplings matter for accuracy once curvature couples distant synapses; the present draft
reports that they do not. Report the result either way.

### A10: Recognised benchmarks (P1)
**Where:** Hellbender (GPU, dataset storage on scratch). Spiking Heidelberg Digits, N-MNIST
or DVS Gesture in a class-incremental protocol, with replay and synaptic-intelligence
baselines alongside EWC, unconstrained and random allocation. 20 seeds as in Fig. 3.

### A11: Check the second-order surrogate (P1, cheap)
**Where:** Hellbender or laptop. Log predicted vs realised loss change per window in
`exp1_continual`; report the correlation (Supplementary). Answers a limitation stated in
the Discussion.

### A12: Flag-fault-tolerant Iceberg gadget (P2)
**Where:** Hellbender. Replace the simple syndrome gadget in `ibm/circuits.py` with the
flag-fault-tolerant version from Self, Benedetti & Amaro (Nat. Phys. 2024); simulate with
routing on FakeBoston and compare accepted-shot quality against the unencoded circuit.
**Blocks:** Q8.

### A13: Full-stack cost comparison (P1)
**Where:** Hellbender + laptop. **Depends:** Q2–Q7 logs, A6. Time (and, if possible,
energy) per allocation including transpilation, queueing and post-processing, against the
classical solvers on one laptop core and one Hellbender node.

---

## Q. Experiments on real quantum hardware

Run every configuration **on at least three different days** and report the spread. For
every job, save: backend, date, calibration snapshot, median two-qubit and readout error on
the qubits used, shots, twirling randomizations, queue time, QPU seconds, wall-clock time
(the runner stores most of this in `results/hardware/*.json`).

Instance sets: `results/instances_n12.json` (40 instances; runner default `--max 20`),
`results/instances_n10.json` (16). Default 4,096 shots.

### Q0: Port the runner to our hardware route (P0, only if needed)
**Where:** laptop + Hellbender. **Depends:** D1. If the hardware is reached through QDMI or a
non-IBM provider, rewrite `ibm/runtime.py` submission (Sampler/Estimator, DD, twirling, ZNE)
for that stack and repeat A2 against its device model. Error-suppression options that only
exist in Qiskit Runtime (twirling, TREX, ZNE) must then be done in our own code.

### Q1: Smoke test and budget (P0)
**Where:** QPU (Heron, then Nighthawk). **Depends:** A2, D1.
`readout_calibration` circuits plus one 12-branch instance at p = 1. Check bit order and
decoded regret against A3. Read `job.usage()` to measure QPU seconds per circuit, then
compute the QPU budget for Q2–Q7 **before** submitting them.

### Q2: Dense 12-branch QAOA sampling, p = 1 and 2, SWAP network (P0)
**Where:** QPU: Nighthawk and Heron.
`python -m ibm.run_hardware --backend <dev> --task qaoa --instances results/instances_n12.json --p {1,2}`
**Fills:** Results, hardware paragraph; Table 2 transpiled counts; Fig. 4 hardware points.
Metrics: p_opt, feasible fraction, best-of-256 regret, transpiled two-qubit gates and depth.
Circuits: 20 instances × 2 depths × 2 backends × 3 days = 240.

### Q3: Same with transpiler routing (P0)
**Where:** QPU: both. Add `--routing auto`. **Fills:** comparison of SWAP network vs
transpiler routing (Results / Supplementary). 240 circuits.

### Q4: Mitigated expectation values (P0)
**Where:** QPU: Heron. `--task estimator` (TREX + ZNE, noise factors 1, 3, 5).
**Depends:** fix A2-c. **Fills:** Fig. 5a hardware panel (unmitigated vs mitigated ⟨H_C⟩
against the noiseless value). ZNE multiplies circuit count by about 3 × twirls; budget from Q1.

### Q5: Error-detected (Iceberg) QAOA (P0)
**Where:** QPU: Heron (mid-circuit measurement for the syndrome option).
`--task iceberg --instances results/instances_n10.json`, plus the unencoded circuit on the
same instances and qubits for a paired comparison. **Fills:** Fig. 5c hardware panel.
Metrics: accepted fraction, p_opt among accepted shots vs unencoded. A2-d suggests the
accepted fraction will be much lower than the routing-free simulation.

### Q6: Collision-model quantum reservoir (P0)
**Where:** QPU: Heron (mid-circuit measurement + reset). `--task reservoir` (4 reservoirs ×
40 steps). **Fills:** Fig. 6 hardware panel. Note: 40 steps gives feature fidelity
(RMSE vs ideal) only; the memory-capacity and task errors in the paper use 1,500-step
sequences, which would need much longer dynamic circuits or many shorter segments.
Decide which quantity the hardware panel shows.

### Q7: Lattice-native 100–120-branch QAOA (P1, decisive for Nature Communications)
**Where:** QPU: Nighthawk (all 120 qubits, p = 1–3; 872 two-qubit gates at p = 2).
**Depends:** A5–A8, Q1. Compare against A6 at equal wall-clock and at equal solution
quality, and against A7. This is the experiment the completion guide says decides the paper.

### Q8: Flag-fault-tolerant error detection on hardware (P2)
**Where:** QPU: Heron. **Depends:** A12. Does the encoded circuit beat the unencoded one
after routing?

---

## W. Writing and admin (no computing; listed so nothing is missed)

| ID | Task | Owner |
|---|---|---|
| W1 | Fill the seven `[TO COMPLETE]` entries: hardware results, hardware Methods, data DOI, code URL/DOI, acknowledgements (incl. IBM Quantum text), author contributions, competing interests; authors and affiliations | Authors |
| W2 | Add hardware panels to Figs 4, 5, 6 in `experiments/make_figures.py` (load `results/hardware/`); keep simulation and hardware visually distinct | Varshith |
| W3 | Check all 46 references against DOIs; nine flagged in completion guide §6 | Authors |
| W4 | Prior-art search for novelty claim (energy-constrained plasticity, QUBO continual learning, QAOA neural network training, quantum neuromorphic) | Authors |
| W5 | Code to a public repository with licence and Zenodo DOI; source-data spreadsheet per figure from `results/*.json` | Varshith |
| W6 | AI-use disclosure (Nature Portfolio policy), cover letter, reporting summary, ORCID | Corresponding author |
| W7 | Keep the paper's boundaries: no claim of quantum advantage, ATP savings or quantum effects in neurons | Everyone |

---

## Suggested order (critical path)

```
D1, D2, D3 ──► A1 ──► A2 ──► A3 ──► Q1 ──► Q2, Q3, Q4, Q5, Q6 (×3 days) ──► W1, W2
                       │
                       └─► Q0 (only if not IBM Runtime)
(if D3 = route 2)  A5 ──► A6, A7 ──► A8 ──► Q7 ──► A13
                   A9, A10, A11 in parallel on Hellbender
```
