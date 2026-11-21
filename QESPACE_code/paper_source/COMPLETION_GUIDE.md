---
title: "Q-ESPACE manuscript: completion guide"
---

# 1. Where the draft stands

The manuscript is a complete first draft in Nature Communications article structure. Every number in it comes from simulations that were actually run with the accompanying code. No hardware experiment has been run, and the draft says so. Entries that only you can supply are marked **[TO COMPLETE: …]** and highlighted in the Word file (red in the LaTeX PDF).

| Item | Status |
|---|---|
| Title, abstract, introduction, results, discussion, methods | Drafted |
| Figures 1 to 6, Tables 1 and 2 | Generated from the code (`experiments/make_figures.py`) |
| Simulation results (learning, sampler, noise, reservoir, resources) | Run; outputs in `results/` |
| IBM Nighthawk / Heron experiments | **Not run.** Circuits and runner written but untested |
| Authors, affiliations, acknowledgements, contributions | To complete |
| Repository DOI, data deposit | To complete |
| Reference check | 46 cited; 9 need checking (Section 6) |

Journal limits and the draft:

| Requirement (Nature Communications article) | Limit | Draft |
|---|---|---|
| Title | about 15 words | 13 |
| Abstract, unreferenced | 150 words | 149 |
| Main text (introduction, results, discussion) | 5,000 words | about 3,400 |
| Display items | 10 | 8 (6 figures, 2 tables) |
| References | about 70 | 46 |
| Figure legends | 350 words each | all under 160 |
| Section order | Introduction, Results, Discussion, Methods | followed |
| Methods, data availability, code availability | required | present |

You have about 1,600 words of room in the main text for the hardware results.

# 2. An honest view of publishability

You asked for a paper that is novel and easy to publish. The formulation is new as far as I can tell: I did not find the four constraints combined into one branch-level quadratic knapsack problem with a slack-free qubit mapping. But I could not search the literature exhaustively, so run a careful prior-art search before claiming novelty (terms: energy-constrained plasticity, synaptic caching, QUBO continual learning, QAOA neural network training, quantum neuromorphic).

It is not yet an easy Nature Communications paper, for three reasons that the simulations themselves expose:

1. **No hardware data.** A paper that names Nighthawk and Heron in the abstract will be expected to show data from them.
2. **No quantum benefit at the sizes tested.** At 12 to 20 branches classical solvers are exact in milliseconds. The draft states this openly; hiding it would not survive review.
3. **Two negative findings.** Coupling terms did not change accuracy, and the quantum reservoir lost to a delay line. These are reported as results. They make the paper more credible and less spectacular.

What would make it competitive is listed in Section 4. If the hardware results turn out modest, the same manuscript fits npj Quantum Information, Communications Physics, PRX Quantum, Quantum Science and Technology, or Neuromorphic Computing and Engineering with little change.

Do not add claims of quantum advantage, ATP savings or quantum effects in neurons to make the paper "stronger". None is supported, and the source document you gave me is explicit on the same point.

# 3. Steps to finish the present draft

## 3.1 Reproduce the simulations (half a day)

1. Unzip `QESPACE_code.zip`, install `requirements.txt`, run `python -m pytest -q` (10 tests should pass).
2. Run the five experiments in the order given in `README.md`, then `make_figures`.
3. Confirm that your numbers match `results/` and the manuscript. Small differences across NumPy versions are possible; if any number changes, update the text.

## 3.2 Validate the IBM code before using hardware time (one day)

The Qiskit files were written without being able to run them. Before any real job:

1. `pip install -r requirements-ibm.txt`
2. `python -m ibm.run_hardware --fake --task qaoa --instances results/instances_n12.json --max 3`
3. Fix whatever the dry run reports. Likely places: option names in `ibm/runtime.py` (they change between qiskit-ibm-runtime releases), the import path of `generate_preset_pass_manager`, and result accessors (`res[k].data.meas.get_counts()`).
4. Check one instance by hand: the regret reported for the fake backend should be close to the simulated value in `results/exp3_noise.json`.
5. Check the current backend list in your IBM account. The draft names `ibm_miami` (Nighthawk, 120 qubits, square lattice) and `ibm_boston` (Heron r3). When Nighthawk was released it did not support dynamic circuits, so the reservoir and syndrome-measurement circuits are assigned to Heron. Confirm whether that has changed.

## 3.3 Hardware experiments to run

| Experiment | Backend | Command (`python -m ibm.run_hardware …`) | Fills |
|---|---|---|---|
| Dense 12-branch sampling, depth 1 and 2 | Nighthawk and Heron | `--task qaoa --instances results/instances_n12.json --p 2` | Results, hardware paragraph; Table 2 transpiled counts |
| Same with transpiler routing instead of the SWAP network | both | add `--routing auto` | Comparison of routing strategies |
| Mitigated expectation values | Heron | `--task estimator` | Fig. 5a hardware panel |
| Error-detected QAOA | Heron | `--task iceberg --instances results/instances_n10.json` | Fig. 5c hardware panel |
| Reservoir, 40 steps | Heron | `--task reservoir` | Fig. 6 hardware panel |
| Lattice-native 100 to 120 branches | Nighthawk | needs larger instances, Section 4 | Scaling result |

Record for the Methods: backend, date, calibration snapshot, median two-qubit and read-out error on the qubits used, shots, twirling randomizations, queue time, QPU seconds, wall-clock time. The runner saves most of this in its JSON.

Run each configuration at least three times on different days. Report the variation.

## 3.4 Fill the placeholders

Search the Word file for "TO COMPLETE". There are seven: hardware results (Results), hardware details (Methods), data deposit, code repository, acknowledgements, author contributions, competing interests. Also replace the author and affiliation lines.

## 3.5 Add hardware panels to the figures

Add your measured points to Figs 4, 5 and 6 in `experiments/make_figures.py` (load the JSON files from `results/hardware/`). Keep simulation and hardware visually distinct and say which is which in the legend.

# 4. What would make the paper strong enough

In rough order of value:

1. **Scale where classical exact methods stop.** Build networks with 100 to 120 branches and lattice-local couplings, run QAOA on all 120 Nighthawk qubits at depth 1 to 3 (872 two-qubit gates at depth 2), and compare with tuned classical solvers on the same instances: Gurobi or CP-SAT, parallel tempering, tabu search, and tensor-network simulation of the same circuit. This is the experiment that decides the paper.
2. **Make the couplings matter.** Extend plasticity from output synapses to the recurrent layer (e-prop or surrogate gradients). Curvature then couples distant synapses and the knapsack-only solution should lose accuracy. If it still does not, say so and present the method as a budgeted-plasticity result with a quantum sampler as one tested solver.
3. **Use recognised benchmarks.** Spiking Heidelberg Digits, N-MNIST or DVS Gesture in a class-incremental protocol, with replay and synaptic-intelligence baselines.
4. **Calibrate the energy.** Either map costs to published ATP estimates with uncertainty ranges, or measure write energy on a neuromorphic or FPGA platform.
5. **Check the surrogate.** Compare the predicted loss change with the realised one per window and report the correlation.
6. **Report full-stack cost.** Time and energy per allocation including compilation and queueing, against the classical solver on a laptop.
7. **Error detection on hardware.** Replace the simple syndrome gadget in `ibm/circuits.py` with the flag-fault-tolerant one from the Iceberg-code paper and measure whether accepted shots beat the unencoded circuit after routing.

# 5. Submission checklist

* Cover letter: the question, the main result, why the journal's readership cares, suggested and excluded reviewers.
* Manuscript as Word or PDF for first submission; figures can stay embedded. For the final version: figures as separate vector files (the code writes PDF and 300 dpi PNG), fonts embedded, 89 mm or 183 mm wide.
* Supplementary Information as a single PDF (compile `supplementary.tex`).
* Source data file for every figure (the JSON files in `results/` can be converted to one spreadsheet per figure).
* Code in a public repository with an archived release (Zenodo DOI) and a licence.
* Reporting summary and editorial policy checklist from the journal's submission system.
* ORCID for the corresponding author; author contributions; competing interests.
* **Use of AI tools.** Nature Portfolio requires that use of a large language model in preparing a manuscript be disclosed (normally in the Methods or acknowledgements), and an AI system cannot be an author. This draft and code were prepared with an AI assistant. You are responsible for checking every statement, equation, number and reference, and for the disclosure.
* IBM Quantum acknowledgement text if you use IBM services.
* Check the current guide for authors on the journal's website before submitting; the limits above were checked in November 2026 but can change.

# 6. References that must be checked

All references should be checked against the DOI before submission. These nine need particular care.

| Reference in the draft | Why |
|---|---|
| Malkin et al., eLife 2024 | From your source document; author list added from memory |
| Schug et al., eLife 2021 | From your source document |
| Blekos et al., Phys. Rep. 2024 | From your source document |
| Koch et al., Nat. Comput. Sci. 2026 | From your source document; first author and title inferred, volume and pages missing |
| Sannia et al., npj Quantum Inf. 2026 | From your source document; published after my reliable knowledge |
| Montañez-Barrera et al., Quantum Sci. Technol. 2024 | Added by me; confirm volume, article number and DOI |
| Self, Benedetti & Amaro, Nat. Phys. 2024 | Added by me; confirm pages and DOI |
| He et al., arXiv:2409.12104 | Added by me; confirm author list and whether it is now published |
| Mujal et al., npj Quantum Inf. 2023 | Added by me; confirm article number and DOI |

Your source document also cited Zhang et al. (npj Quantum Inf. 2026), Chen & Zhang (PRX Quantum 2026), a Gaussian-boson-sampler reservoir paper and arXiv:2607.18552. I left them out of the draft because I could not confirm them; add them back after checking if you want them. The file `references.bib` contains BibTeX entries for all cited works.

# 7. What changed relative to your Q-ESPACE document

* The decision variable is now a dendritic branch, not a single synapse, so a network with hundreds of synapses gives a problem of 12 to 120 qubits.
* The slack-variable encoding is replaced by unbalanced penalization (no extra qubits); the slack version is kept for comparison.
* The QAOA angles are trained once and reused, so no variational loop runs during learning.
* The network is a recurrent spiking network with calcium-dependent release and vesicle depletion, five tasks instead of two, 20 seeds, paired statistics and ablations.
* The delayed-density-matrix reservoir, which cannot be built, is replaced by a collision-model reservoir that can.
* Noise simulation, read-out mitigation, zero-noise extrapolation and error detection are added, with hardware circuits.
* The scientific boundaries of your document are kept: classical synapses, no claim of quantum advantage, negative controls reported.
