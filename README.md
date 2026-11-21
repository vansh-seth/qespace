# Q-ESPACE paper: completion tasks

Task list for completing the manuscript *Quantum-assisted allocation of metabolically
constrained synaptic plasticity for continual learning in spiking networks*.

**The full list is in [TASKS.md](TASKS.md).**

## Summary

| Group | Where | Tasks | Needed for |
|---|---|---|---|
| D: decisions | people | D1 hardware access, D2 Hellbender allocation, D3 scope | everything |
| A: simulations | **Hellbender** | A1 reproduce, A2 validate IBM code, A3 noisy predictions (P0); A5–A11, A13 scaling, baselines, extensions (P1); A4, A12 (P2) | figures, tables, scaling result |
| Q: hardware | **real QPU** | Q1 smoke test, Q2–Q6 12-branch sampling, routing, mitigation, error detection, reservoir (P0); Q7 120-branch Nighthawk run (P1); Q8 (P2) | hardware paragraph, Figs 4–6 hardware panels, Table 2 |
| W: writing | authors | placeholders, figure panels, references, deposit, disclosure | submission |

P0 = needed to finish the present draft. P1 = needed for a competitive Nature
Communications submission. P2 = optional.

## Where things stand (21 Nov 2026)

* The manuscript, Overleaf source, code package and completion guide (version of
  14 Nov 2026) have been reviewed in full.
* Every number currently in the draft comes from classical simulations that were already
  run with the accompanying code (`qespace/`, `experiments/`); outputs are in `results/`.
* **No quantum-hardware job has been run.** The Qiskit circuits and runner in `ibm/` are
  written but untested; `A2` is the first thing to do before any QPU time is spent.
* A collaborator has agreed to help get access to real quantum hardware and run the
  QPU experiments (section **Q** in TASKS.md); see task **D1**.

## Scope

Two routes (task **D3**):

1. **Finish the present draft**: P0 tasks only — reproduce the simulations, validate the
   IBM code, run the hardware experiments at 10–12 branches, fill the placeholders.
2. **Nature Communications**: additionally run the 100–120-branch scaling experiment
   against tuned classical solvers (A5–A8, Q7), which the completion guide calls "the
   experiment that decides the paper."

The paper source and code are not in this repository; keep them in a private repository
until submission.
