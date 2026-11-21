"""Experiment 5: hardware resource estimates and lattice-native truncation.

(a) analytic two-qubit gate counts for dense (SWAP-network) and lattice-native
    QAOA circuits, compared with a 5,000 two-qubit-gate budget;
(b) cost of dropping all couplings that are not nearest neighbours on a square
    lattice: solve the truncated problem exactly, score it on the full objective.

  python -m experiments.exp5_resources --out results
"""
import argparse, json
from pathlib import Path
import numpy as np
from qespace import qubo as Q

GRIDS = {12: (3, 4), 16: (4, 4), 20: (4, 5)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); args = ap.parse_args()
    out = Path(args.out); res = {"counts": [], "truncation": {}}
    for n, (r, c) in [(12, (3, 4)), (16, (4, 4)), (20, (4, 5)), (36, (6, 6)), (64, (8, 8)), (100, (10, 10)), (120, (10, 12))]:
        for p in (1, 2, 4):
            res["counts"].append(dict(n=n, p=p, swap=Q.gate_counts(n, p, "swap"), grid=Q.gate_counts(n, p, "grid", r, c)))
    for n, (r, c) in GRIDS.items():
        f = out / f"instances_n{n}.json"
        if not f.exists():
            continue
        reg_grid, reg_diag, frac = [], [], []
        for d in json.load(open(f)):
            a, J, cost, B = np.array(d["a"]), np.array(d["J"]), np.array(d["cost"]), d["B"]
            _, vs = Q.exact_feasible(a, J, cost, B)
            if vs >= 0:
                continue
            Jt = Q.truncate_to_grid(J, r, c)
            ug, _ = Q.exact_feasible(a, Jt, cost, B); ud, _ = Q.exact_feasible(a, 0 * J, cost, B)
            reg_grid.append((Q.value(ug, a, J) - vs) / abs(vs)); reg_diag.append((Q.value(ud, a, J) - vs) / abs(vs))
            frac.append(np.abs(Jt).sum() / (np.abs(J).sum() + 1e-15))
        res["truncation"][n] = dict(grid=[r, c], regret_grid=float(np.mean(reg_grid)), regret_diag=float(np.mean(reg_diag)),
                                    coupling_weight_kept=float(np.mean(frac)), instances=len(reg_grid))
        print(n, res["truncation"][n])
    for row in res["counts"]:
        if row["p"] == 2:
            print(row)
    json.dump(res, open(out / "exp5_resources.json", "w"))


if __name__ == "__main__":
    main()
