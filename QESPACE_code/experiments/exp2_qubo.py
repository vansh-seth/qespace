"""Experiment 2: solver benchmark on QUBO instances extracted from learning traces.

Instances come from the spiking network (not from a synthetic generator).
Development instances (seed 991) are used for angle training; test instances
come from different seeds. QAOA is simulated exactly (state vector), so success
probabilities are exact, not sampled.

  python -m experiments.exp2_qubo --out results
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from qespace.snn import NetConfig
from qespace import qubo as Q, qaoa
from experiments.exp1_continual import run_seed, penalty_weights


def get_instances(n, seeds, per_seed=20):
    cfg = NetConfig(n_groups=n, n_hid=8 * n)
    out = []
    for s in seeds:
        dump = []
        run_seed(s, cfg, allocators=["exact"], dump=dump)
        good = [d for d in dump if d["B"] > 0]
        good = [d for d in good if Q.exact_feasible(np.array(d["a"]), np.array(d["J"]), np.array(d["cost"]), d["B"])[1] < 0]
        out += [good[i] for i in np.linspace(0, len(good) - 1, per_seed).astype(int)]
    return out


def prepare(d, encoding="unbalanced"):
    a = np.array(d["a"]); J = np.array(d["J"]); c = np.array(d["cost"]); B = d["B"]; n = len(a)
    l1, l2 = penalty_weights(a, c)
    if encoding == "unbalanced":
        a2, J2, _ = Q.unbalanced_qubo(a, J, c, B, l1, l2)
    else:
        a2, J2, _ = Q.slack_qubo(a, J, c, B, P=l2 * 5)
    N = len(a2)
    bits = Q.all_bits(N)[:, :n]
    v = bits @ a + np.einsum("bi,ij,bj->b", bits, np.triu(J, 1), bits)
    feas = bits @ c <= B + 1e-9
    vf = np.where(feas, v, np.inf); vs = vf.min()
    return dict(E=qaoa.normalised_energies(a2, J2), opt=vf <= vs + 1e-12 * (1 + abs(vs)),
                near=vf <= 0.99 * vs, feas=feas, v=v, vs=vs, N=N, a=a, J=J, c=c, B=B)


def qaoa_stats(P, ang, shots=256, rng=None):
    p = len(ang) // 2
    pr = qaoa.qaoa_probs(P["E"], ang[:p], ang[p:]) if p else np.full(len(P["E"]), 1.0 / len(P["E"]))
    p_opt = float(pr[P["opt"]].sum()); p_near = float(pr[P["near"]].sum()); p_feas = float(pr[P["feas"]].sum())
    # expected regret of best-feasible-of-`shots` (Monte Carlo over 20 repetitions)
    reg = []
    for _ in range(20):
        idx = rng.choice(len(pr), shots, p=pr / pr.sum())
        vv = np.where(P["feas"][idx], P["v"][idx], 0.0)
        reg.append((min(vv.min(), 0.0) - P["vs"]) / abs(P["vs"]))
    return dict(p_opt=p_opt, p_near=p_near, p_feas=p_feas, regret=float(np.mean(reg)),
                shots99=float(np.log(0.01) / np.log1p(-min(p_opt, 1 - 1e-12))) if p_opt > 0 else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[12, 16, 20])
    ap.add_argument("--pmax", type=int, default=4)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    out = Path(args.out); res = {"sizes": args.sizes, "qaoa": {}, "classical": {}, "angles": {}}

    # ---- angle training on development instances at the smallest size, then transfer
    dev = [prepare(d) for d in get_instances(args.sizes[0], [991], per_seed=12)]
    angles = {0: np.array([])}; x = None
    for p in range(1, args.pmax + 1):
        x, f = qaoa.train_angles([d["E"] for d in dev], p, seed=p, x0=qaoa.interp_init(x) if x is not None else None)
        angles[p] = x; print("p", p, "angles", np.round(x, 3), "dev <E>", round(f, 3), flush=True)
    res["angles"] = {p: a.tolist() for p, a in angles.items()}

    for n in args.sizes:
        test = get_instances(n, [1, 2], per_seed=20)
        json.dump(test, open(out / f"instances_n{n}.json", "w"))
        # classical solvers
        rows = []
        for d in test:
            a, J, c, B = np.array(d["a"]), np.array(d["J"]), np.array(d["cost"]), d["B"]
            t0 = time.perf_counter(); _, vs = Q.exact_feasible(a, J, c, B); te = time.perf_counter() - t0
            r = dict(exact_s=te)
            for name, fn in (("greedy", lambda: Q.greedy_feasible(a, J, c, B)),
                             ("diag", lambda: Q.knapsack_diag(a, J, c, B)),
                             ("anneal", lambda: Q.simulated_annealing(a, J, c, B, rng, iters=2000))):
                t0 = time.perf_counter(); _, v = fn(); r[name + "_s"] = time.perf_counter() - t0
                r[name + "_regret"] = (v - vs) / abs(vs)
            rows.append(r)
        res["classical"][n] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for k in ("greedy", "diag", "anneal"):
            res["classical"][n][k + "_hit"] = float(np.mean([r[k + "_regret"] < 1e-9 for r in rows]))
        print(n, "classical", {k: round(v, 5) for k, v in res["classical"][n].items()}, flush=True)
        # QAOA (transferred angles), unbalanced encoding
        P = [prepare(d) for d in test]
        res["qaoa"][n] = {}
        for p in range(0, args.pmax + 1):
            st = [qaoa_stats(q, angles[p], rng=rng) for q in P]
            res["qaoa"][n][p] = {k: [s[k] for s in st] for k in st[0]}
            m = {k: float(np.mean(v)) for k, v in res["qaoa"][n][p].items() if k != "shots99"}
            m["shots99_median"] = float(np.median(res["qaoa"][n][p]["shots99"]))
            print(n, "QAOA p", p, {k: round(v, 5) for k, v in m.items()}, flush=True)
        if n == args.sizes[0]:   # slack encoding comparison at the smallest size
            Ps = [prepare(d, "slack") for d in test]
            xs = None; res["slack"] = {"N": int(np.mean([q["N"] for q in Ps]))}
            for p in (1, 2):
                xs, _ = qaoa.train_angles([prepare(d, "slack")["E"] for d in get_instances(n, [991], per_seed=6)], p,
                                          seed=p, restarts=2, maxiter=200, x0=qaoa.interp_init(xs) if xs is not None else None)
                st = [qaoa_stats(q, xs, rng=rng) for q in Ps]
                res["slack"][p] = {k: float(np.mean([s[k] for s in st])) for k in ("p_opt", "p_feas", "regret")}
                print("slack p", p, res["slack"][p], flush=True)
    json.dump(res, open(out / "exp2_qubo.json", "w"))


if __name__ == "__main__":
    main()
