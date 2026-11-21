"""Experiment 1: continual learning under an ATP-limited plasticity budget.

Compares allocators that decide, once per learning window, which dendritic
branches commit their three-factor candidate update. All allocators see the
same pre-generated spike/release data (paired design).

  python -m experiments.exp1_continual --seeds 20 --out results
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from qespace.snn import NetConfig, Network, Readout, make_task, encode, window_quadratic
from qespace import qubo as Q
from qespace import qaoa

ALLOCATORS = ["dense", "dense_ewc", "random", "diag", "greedy", "anneal", "exact", "qaoa", "uniform"]
ABLATIONS = {"full": {}, "no_fisher": dict(lam_F=0.0), "no_risk": dict(lam_U=0.0),
             "no_pair": dict(use_pair=False), "no_energy": dict(dense=True)}


def penalty_weights(a, cost):
    """Unbalanced-penalisation weights from instance scale (fixed rule, no tuning per instance)."""
    s = np.abs(a).mean() + 1e-12
    return 0.5 * s, 0.01 * s      # selected once on development seed 991 (see exp2 --tune)


def allocate(name, a, J, cost, B, rng, angles=None, shots=256):
    n = len(a)
    if name == "dense":
        u = np.ones(n, int); return u, Q.value(u, a, J), {}
    if name == "random":
        u, v = Q.random_feasible(a, J, cost, B, rng); return u, v, {}
    if name == "diag":
        u, v = Q.knapsack_diag(a, J, cost, B); return u, v, {}
    if name == "greedy":
        u, v = Q.greedy_feasible(a, J, cost, B); return u, v, {}
    if name == "anneal":
        u, v = Q.simulated_annealing(a, J, cost, B, rng, iters=1500); return u, v, {}
    if name == "exact":
        u, v = Q.exact_feasible(a, J, cost, B); return u, v, {}
    if name == "uniform":                         # uniform sampler, same shots and post-processing as QAOA
        bits = rng.integers(0, 2, (shots, n))
        u, v, ff = Q.best_of_samples(bits, a, J, cost, B); return u, v, dict(feasible=ff)
    if name == "qaoa":
        l1, l2 = penalty_weights(a, cost)
        a2, J2, _ = Q.unbalanced_qubo(a, J, cost, B, l1, l2)
        p = len(angles) // 2
        probs = qaoa.qaoa_probs(qaoa.normalised_energies(a2, J2), angles[:p], angles[p:])
        bits = qaoa.sample_bits(probs, shots, rng)
        u, v, ff = Q.best_of_samples(bits, a, J, cost, B); return u, v, dict(feasible=ff)
    raise ValueError(name)


def run_seed(seed, cfg, n_tasks=5, windows=60, W=10, n_test=200, allocators=ALLOCATORS,
             ablation="full", angles=None, eta=1.0, lam_F=8.0, lam_F_ewc=2.0, dump=None):
    rng = np.random.default_rng(seed)
    net = Network(cfg, rng)
    tasks = [make_task(cfg, rng) for _ in range(n_tasks)]
    train = [[encode(net, t, W, rng) for _ in range(windows)] for t in tasks]
    test = [encode(net, t, n_test, rng)[:2] for t in tasks]
    w0 = rng.normal(0, 0.01, cfg.n_hid)
    kw = dict(ABLATIONS[ablation]); force_dense = kw.pop("dense", False)
    out = {}
    for name in allocators:
        arng = np.random.default_rng(seed * 1000 + 17)
        ro = Readout(w=w0.copy()); A = cfg.A_max
        acc = np.zeros((n_tasks, n_tasks)); upd = 0.0; regret = []; nsel = []; t0 = time.perf_counter()
        fisher_sum = np.zeros(cfg.n_hid)
        for k in range(n_tasks):
            for (x, y, p_mean, n_rel, n_sp) in train[k]:
                delta, a, J, db = window_quadratic(net, ro, x, y, p_mean, eta=eta, **{**dict(lam_F=lam_F), **kw})
                B = int(np.floor(A))
                if name == "dense_ewc" and ro.fisher is not None:   # standard EWC, no budget
                    delta = delta - eta * lam_F_ewc * ro.fisher * (ro.w - ro.anchor)
                if force_dense or name in ("dense", "dense_ewc"):
                    u = np.ones(cfg.n_groups, int); v = Q.value(u, a, J)
                else:
                    u, v, _ = allocate(name, a, J, net.cost, B, arng, angles)
                    assert net.cost @ u <= B
                    _, vstar = Q.exact_feasible(a, J, net.cost, B)
                    regret.append((v - vstar, abs(vstar)))
                    if dump is not None and name == "exact" and len(dump) < 400:
                        dump.append(dict(a=a.tolist(), J=J.tolist(), cost=net.cost.tolist(), B=B, seed=seed))
                spent = float(net.cost @ u)
                ro.w = ro.w + delta * (net.G.T @ u); ro.b += db
                upd += spent; nsel.append(int(u.sum()))
                if not (force_dense or name in ("dense", "dense_ewc")):
                    A = min(cfg.A_max, A - spent + cfg.atp_refill)
            for j in range(n_tasks):
                acc[k, j] = ro.accuracy(*test[j])
            xs = np.concatenate([t[0] for t in train[k]])
            pi = ro.prob(xs)
            fisher_sum += np.mean((pi * (1 - pi))[:, None] * xs ** 2, 0)
            ro.fisher = fisher_sum.copy(); ro.anchor = ro.w.copy()
        final = acc[-1]
        forget = float(np.mean([max(acc[k:, j].max() - final[j], 0) for j, k in zip(range(n_tasks - 1), range(n_tasks - 1))]))
        out[name] = dict(final_acc=float(final.mean()), forgetting=forget, update_cost=upd,
                         learn_acc=float(np.mean(np.diag(acc))),
                         mean_regret=float(np.sum([r[0] for r in regret]) / (np.sum([r[1] for r in regret]) + 1e-15)) if regret else None,
                         hit_opt=float(np.mean([r[0] < 1e-9 * (1 + r[1]) for r in regret])) if regret else None,
                         mean_selected=float(np.mean(nsel)), cpu_s=time.perf_counter() - t0,
                         acc_matrix=acc.tolist())
    return out


def train_transfer_angles(cfg, p, n_inst=12, seed=991):
    """Train one angle schedule on instances from a held-out development seed."""
    dump = []
    run_seed(seed, cfg, allocators=["exact"], dump=dump)
    idx = np.linspace(0, len(dump) - 1, n_inst).astype(int)
    Es = []
    for i in idx:
        d = dump[i]; a = np.array(d["a"]); J = np.array(d["J"]); c = np.array(d["cost"])
        l1, l2 = penalty_weights(a, c)
        a2, J2, _ = Q.unbalanced_qubo(a, J, c, d["B"], l1, l2)
        Es.append(qaoa.normalised_energies(a2, J2))
    x = None
    for q in range(1, p + 1):                      # layer-wise training with interpolation init
        x, f = qaoa.train_angles(Es, q, seed=seed, x0=qaoa.interp_init(x) if x is not None else None)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--groups", type=int, default=12)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = NetConfig(n_groups=args.groups)
    out = Path(args.out); out.mkdir(exist_ok=True)
    angles = train_transfer_angles(cfg, args.p)
    print("transfer angles", np.round(angles, 4))
    res = {"angles": angles.tolist(), "p": args.p, "config": cfg.__dict__, "seeds": [], "ablation": {}}
    dump = []
    for s in range(args.seeds):
        r = run_seed(s, cfg, angles=angles, dump=dump)
        res["seeds"].append(r)
        print(s, {k: round(v["final_acc"], 3) for k, v in r.items()}, flush=True)
    for ab in ABLATIONS:
        res["ablation"][ab] = [run_seed(s, cfg, allocators=["exact"], ablation=ab)["exact"] for s in range(args.seeds)]
    (out / "exp1_continual.json").write_text(json.dumps(res))
    (out / f"instances_n{args.groups}.json").write_text(json.dumps(dump))
    print("\nallocator        final_acc  forgetting  update_cost  regret   hit_opt")
    for name in ALLOCATORS:
        g = lambda key: np.array([r[name][key] for r in res["seeds"] if r[name][key] is not None], float)
        fa, fo, uc = g("final_acc"), g("forgetting"), g("update_cost")
        rg, ho = g("mean_regret"), g("hit_opt")
        print(f"{name:14s} {fa.mean():.3f}±{fa.std(ddof=1)/np.sqrt(len(fa)):.3f}  {fo.mean():.3f}  {uc.mean():8.1f}"
              + (f"  {rg.mean():.4f}  {ho.mean():.3f}" if len(rg) else ""))
    for ab, rows in res["ablation"].items():
        print(f"ablation {ab:10s} acc {np.mean([r['final_acc'] for r in rows]):.3f} "
              f"forget {np.mean([r['forgetting'] for r in rows]):.3f} cost {np.mean([r['update_cost'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
