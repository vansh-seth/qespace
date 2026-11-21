"""Experiment 3: device-noise simulation, error mitigation and error detection.

(a) expectation values: raw vs read-out mitigation vs zero-noise extrapolation;
(b) samples: probability of the optimal feasible mask, feasible fraction;
(c) Iceberg [[k+2,k,2]] error-detected QAOA vs the unencoded circuit.

Noise is a simulated depolarising + read-out model, NOT hardware data.

  python -m experiments.exp3_noise --out results
"""
import argparse, json
from pathlib import Path
import numpy as np
from qespace import qubo as Q, noise as NZ, iceberg
from experiments.exp1_continual import penalty_weights
from experiments.exp2_qubo import get_instances


def ising_of(d):
    a = np.array(d["a"]); J = np.array(d["J"]); c = np.array(d["cost"]); B = d["B"]
    l1, l2 = penalty_weights(a, c)
    a2, J2, _ = Q.unbalanced_qubo(a, J, c, B, l1, l2)
    h, K, _ = Q.qubo_to_ising(a2, J2); s = Q.ising_scale(h, K)
    bits = Q.all_bits(len(a)); v = Q.all_values(a, J); feas = bits @ c <= B + 1e-9
    vf = np.where(feas, v, np.inf)
    return h / s, K / s, vf <= vf.min() + 1e-12 * (1 + abs(vf.min())), feas, (a, J, c, B, vf.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--inst", type=int, default=16)
    ap.add_argument("--traj", type=int, default=150)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    ang = json.load(open(Path(args.out) / "exp1_continual.json"))["angles"]   # depth-2 angles of the learning loop
    g, b = ang[:2], ang[2:]
    inst = get_instances(args.n, [3], per_seed=args.inst)
    res = {"n": args.n, "p2": [], "angles": ang, "shots": args.shots}
    for p2 in (1e-3, 2.5e-3, 5e-3):
        nm = NZ.NoiseModel(p2=p2, swap_overhead=3.0)
        rows = []
        for d in inst:
            h, K, opt, feas, (a, J, c, B, vs) = ising_of(d)
            build = lambda s: NZ.qaoa_circuit(s, h, K, g, b)
            pid, n2q = NZ.averaged_probs(build, args.n, None, rng)
            z = 1 - 2 * Q.all_bits(args.n)
            Eall = z @ h + np.einsum("bi,ij,bj->b", z, np.triu(K, 1), z)
            E_ideal = float(pid @ Eall)
            r = dict(E_ideal=E_ideal, popt_ideal=float(pid[opt].sum()), pfeas_ideal=float(pid[feas].sum()), n2q=n2q)
            raw, ro = [], []
            for lam in (1, 3, 5):
                pn, _ = NZ.averaged_probs(build, args.n, nm.scaled(lam), rng, args.traj)
                bits = NZ.sample_with_readout(pn, args.shots, nm, rng)
                raw.append(NZ.ising_expectation(bits, h, K))
                ro.append(NZ.ising_expectation(bits, h, K, readout=(nm.p01, nm.p10)))
                if lam == 1:
                    idx = bits @ (1 << np.arange(args.n))
                    r["popt_noisy"] = float(opt[idx].mean()); r["pfeas_noisy"] = float(feas[idx].mean())
                    sub = bits[:256]
                    _, v, _ = Q.best_of_samples(sub, a, J, c, B)
                    r["regret256_noisy"] = float((v - vs) / abs(vs))
            r.update(E_raw=raw[0], E_ro=ro[0],
                     E_zne_lin=NZ.extrapolate([1, 3, 5], ro, "linear"),
                     E_zne_quad=NZ.extrapolate([1, 3, 5], ro, "quadratic"),
                     E_zne_exp=NZ.extrapolate([1, 3, 5], ro, "exponential"),
                     E_lams=ro)
            rows.append(r)
        res["p2"].append(dict(p2=p2, rows=rows))
        E = lambda k: np.mean([abs(r[k] - r["E_ideal"]) / abs(r["E_ideal"]) for r in rows])
        print(f"p2={p2:g} 2q-gates={rows[0]['n2q']}  rel.err: raw {E('E_raw'):.3f} readout {E('E_ro'):.3f} "
              f"ZNE-lin {E('E_zne_lin'):.3f} quad {E('E_zne_quad'):.3f} exp {E('E_zne_exp'):.3f} | "
              f"p_opt ideal {np.mean([r['popt_ideal'] for r in rows]):.4f} noisy {np.mean([r['popt_noisy'] for r in rows]):.4f} "
              f"regret256 {np.mean([r['regret256_noisy'] for r in rows]):.4f}", flush=True)

    # ---------------- Iceberg error detection (same per-gate noise, connectivity-agnostic)
    res["iceberg"] = []
    for p2 in (1e-3, 2.5e-3, 5e-3, 1e-2):
        nm = NZ.NoiseModel(p2=p2, swap_overhead=1.0)
        rows = []
        for d in inst[:10]:
            h, K, opt, feas, _ = ising_of(d); k = args.n
            pu, n2u = NZ.averaged_probs(lambda s: NZ.qaoa_circuit(s, h, K, g, b), k, nm, rng, args.traj)
            bu = NZ.sample_with_readout(pu, args.shots, nm, rng)
            pe, n2e = NZ.averaged_probs(lambda s: iceberg.iceberg_qaoa(s, h, K, g, b), k + 2, nm, rng, args.traj)
            be = NZ.sample_with_readout(pe, args.shots, nm, rng)
            logical, ok = iceberg.decode(be)
            pid, _ = NZ.averaged_probs(lambda s: NZ.qaoa_circuit(s, h, K, g, b), k, None, rng)
            w = 1 << np.arange(k)
            rows.append(dict(popt_ideal=float(pid[opt].sum()), popt_unenc=float(opt[bu @ w].mean()),
                             popt_enc=float(opt[logical @ w].mean()), accept=float(ok.mean()),
                             n2q_unenc=n2u, n2q_enc=n2e))
        m = {k2: float(np.mean([r[k2] for r in rows])) for k2 in rows[0]}
        res["iceberg"].append(dict(p2=p2, rows=rows, mean=m))
        print("iceberg p2", p2, {k2: round(v, 4) for k2, v in m.items()}, flush=True)
    json.dump(res, open(Path(args.out) / "exp3_noise.json", "w"))


if __name__ == "__main__":
    main()
