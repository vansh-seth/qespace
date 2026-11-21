# Q-ESPACE: completion tasks

Draft reviewed: *Quantum-assisted allocation of metabolically constrained synaptic
plasticity for continual learning in spiking networks* (manuscript, Overleaf source, code
package and completion guide, version of 14 Nov 2026).

**Where:** Hellbender = classical simulation/HPC. QPU = real quantum hardware. Laptop =
light analysis/writing.
**Priority:** P0 = needed to finish the current draft. P1 = needed for a competitive
Nature Communications submission. P2 = optional.

---

## D. Decisions needed first

- **D1** Confirm quantum hardware access (devices, plan, QPU minutes, dynamic circuits).
  *In progress — collaborator handling QPU access.*
- **D2** Hellbender allocation (account, partitions, scratch, Gurobi licence).
- **D3** Scope: finish the current draft (P0 only) vs. push for Nature Communications
  (adds the 100–120-branch scaling experiment, A5–A8, Q7 — "the experiment that decides
  the paper").

## A. Simulations — Hellbender

- **A1** Reproduce all reported results (tests + 5 experiment scripts). P0.
- **A2** Dry-run the IBM/Qiskit code on fake backends before any real QPU time. P0.
- **A3** Simulate hardware circuits with device noise models to predict QPU results. P0.
- **A4** Re-run noise study with device-calibrated noise. P2.
- **A5–A8** 100–120-branch instances, classical baselines, tensor-network check, angle
  transfer — only needed for the Nat. Comms scaling result. P1.
- **A9** Extend plasticity to the recurrent layer; retest whether couplings matter. P1.
- **A10** Run on recognized benchmarks (SHD, N-MNIST, DVS Gesture). P1.
- **A11** Sanity-check the surrogate loss model. P1.
- **A12** Better error-detection gadget. P2.
- **A13** Full-stack cost comparison, quantum vs. classical. P1.

## Q. Experiments — real QPU

Run each ≥3 different days; log backend, calibration, error rates, shots, QPU time.

- **Q0** Port the runner, only if hardware isn't reached via IBM Runtime.
- **Q1** Smoke test + QPU budget check. P0.
- **Q2** Dense 12-branch QAOA sampling (depth 1–2, SWAP network). P0.
- **Q3** Same, with transpiler auto-routing. P0.
- **Q4** Mitigated expectation values (ZNE + readout correction). P0.
- **Q5** Error-detected (Iceberg code) QAOA. P0.
- **Q6** Collision-model quantum reservoir. P0.
- **Q7** Lattice-native 120-branch QAOA on full Nighthawk — decisive for Nat. Comms. P1.
- **Q8** Flag-fault-tolerant error detection on hardware. P2.

## W. Writing / admin — no computing

Fill 7 `[TO COMPLETE]` placeholders, add hardware panels to Figs 4–6, check 9 flagged
references, run a prior-art search, deposit code with a DOI, AI-use disclosure, cover
letter.

---

## Critical path

```
D1, D2, D3 → A1 → A2 → A3 → Q1 → Q2–Q6 (×3 days) → W
                                 (Nat. Comms route) A5–A8 → Q7 → A13
```
