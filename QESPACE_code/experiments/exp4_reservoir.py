"""Experiment 4: collision-model quantum reservoir vs matched classical baselines.

Tasks (all with 12 read-out features per model):
  memory    linear short-term memory capacity  sum_tau r^2(s_{t-tau})
  product   nonlinear delayed product          s_t * s_{t-tau}
  credit    three-factor eligibility signal    m_t * sum_k lam^k s_{t-1-k}

Hyper-parameters of every model are selected on a validation split with the
same grid size (6 settings each).

  python -m experiments.exp4_reservoir --seeds 10 --out results
"""
import argparse, json
from pathlib import Path
import numpy as np
from qespace import reservoir as R

T, WASH, NTR, NVAL = 1500, 50, 1000, 800
Q_GRID = [(0.4, 0.7, 1.0, 1.3), (0.6, 1.0, 1.4, 1.8), (1.0, 1.5, 2.0, 2.5),
          (0.3, 0.5, 0.7, 0.9), (0.8, 0.8, 1.6, 1.6), (0.5, 1.0, 2.0, 3.0)]
E_GRID = [(0.5, 0.5), (0.9, 0.3), (0.9, 0.6), (0.9, 1.0), (1.1, 0.6), (0.7, 1.0)]


def val_score(F, y):
    return R.ridge_eval(F[:NTR], y[:NTR], WASH, NVAL)[0]


def select(feature_fns, targets):
    """Pick the setting with the lowest mean validation NMSE, return its test scores."""
    feats = [f() for f in feature_fns]
    best = int(np.argmin([np.mean([val_score(F, y) for y in targets]) for F in feats]))
    return feats[best], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    taus_mem = list(range(0, 9)); taus_prod = [1, 2, 3, 4]
    res = {"models": ["quantum", "quantum_shots", "quantum_dephased", "esn", "delay", "oracle"],
           "taus_mem": taus_mem, "taus_prod": taus_prod, "seeds": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        s = rng.uniform(0, 1, T); m = rng.uniform(0, 1, T)
        sh = lambda k: np.r_[np.zeros(k), s[:T - k]]
        targets = dict(memory=[sh(k) for k in taus_mem], product=[s * sh(k) for k in taus_prod],
                       credit=[R.eligibility_target(s, m, 0.8, 1)])
        out = {}
        for task, ys in targets.items():
            inp = np.column_stack([s, m]) if task == "credit" else s[:, None]
            Fq, bq = select([lambda t=t: R.quantum_features(inp, taus=t, seed=seed) for t in Q_GRID], ys)
            models = {
                "quantum": Fq,
                "quantum_shots": R.quantum_features(inp, taus=Q_GRID[bq], seed=seed, shots=2048, rng=rng),
                "quantum_dephased": R.quantum_features(inp, taus=Q_GRID[bq], seed=seed, dephase=0.05),
                "esn": select([lambda g=g: R.esn_features(inp, 12, g[0], g[1], seed=seed) for g in E_GRID], ys)[0],
                "delay": R.delay_features(inp, taps=12 // inp.shape[1]),
            }
            # task-aware oracle: explicit delays plus the exact nonlinear terms
            d = R.delay_features(inp, taps=12 // inp.shape[1])
            if task == "product":
                models["oracle"] = np.column_stack([d] + [s * sh(k) for k in taus_prod])
            elif task == "credit":
                models["oracle"] = np.column_stack([d] + [m * sh(k) for k in range(1, 12)])
            else:
                models["oracle"] = d
            out[task] = {name: [R.ridge_eval(F, y, WASH, NTR) for y in ys] for name, F in models.items()}
        res["seeds"].append(out)
        print(seed, {t: {k: round(float(np.mean([x[0] for x in v])), 3) for k, v in out[t].items()} for t in out}, flush=True)
    json.dump(res, open(Path(args.out) / "exp4_reservoir.json", "w"))
    print("\nmean over seeds")
    for task in ("memory", "product", "credit"):
        for name in res["models"]:
            nm = np.mean([[x[0] for x in sd[task][name]] for sd in res["seeds"]])
            mc = np.mean([np.sum([x[1] for x in sd[task][name]]) for sd in res["seeds"]])
            print(f"{task:8s} {name:17s} NMSE {nm:.3f}   sum r2 {mc:.2f}")


if __name__ == "__main__":
    main()
