"""Run Q-ESPACE circuits on IBM Quantum hardware (or a local fake backend).

Examples
  # local dry run on a noisy Heron snapshot (no IBM account needed)
  python -m ibm.run_hardware --fake --task qaoa --instances results/instances_n12.json --max 5

  # Nighthawk, dense 12-variable instances through the SWAP network
  python -m ibm.run_hardware --backend ibm_miami --task qaoa --instances results/instances_n12.json

  # Heron r3, mitigated expectation values (TREX + ZNE)
  python -m ibm.run_hardware --backend ibm_boston --task estimator --instances results/instances_n12.json

  # error-detected QAOA and the reservoir (needs mid-circuit measurement -> Heron)
  python -m ibm.run_hardware --backend ibm_boston --task iceberg --instances results/instances_n10.json
  python -m ibm.run_hardware --backend ibm_boston --task reservoir

Every run writes a JSON with raw counts, decoded metrics, transpiled gate counts,
job ids and timing, i.e. what the manuscript's hardware tables need.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from qespace import qubo as Q, noise as NZ, iceberg as IB, reservoir as RS
from experiments.exp1_continual import penalty_weights
from . import circuits as C, runtime as RT


def load_instance(d):
    a, J, c, B = np.array(d["a"]), np.array(d["J"]), np.array(d["cost"]), int(d["B"])
    l1, l2 = penalty_weights(a, c)
    a2, J2, _ = Q.unbalanced_qubo(a, J, c, B, l1, l2)
    h, K, _ = Q.qubo_to_ising(a2, J2); s = Q.ising_scale(h, K)
    _, vstar = Q.exact_feasible(a, J, c, B) if len(a) <= 22 else (None, None)
    return dict(a=a, J=J, c=c, B=B, h=h / s, K=K / s, vstar=vstar)


def score(bits, I):
    u, v, ff = Q.best_of_samples(bits, I["a"], I["J"], I["c"], I["B"])
    out = dict(best_value=v, feasible_fraction=ff, shots=int(len(bits)), mask=u.tolist())
    if I["vstar"] is not None:
        out["regret"] = float((v - I["vstar"]) / abs(I["vstar"]))
        vals = bits @ I["a"] + np.einsum("bi,ij,bj->b", bits, np.triu(I["J"], 1), bits)
        feas = bits @ I["c"] <= I["B"]
        out["p_opt"] = float(np.mean(feas & (vals <= I["vstar"] + 1e-12 * (1 + abs(I["vstar"])))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qaoa", "estimator", "iceberg", "reservoir"], default="qaoa")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--instances", default="results/instances_n12.json")
    ap.add_argument("--angles", default="results/exp2_qubo.json")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--routing", choices=["swap", "auto"], default="swap")
    ap.add_argument("--fractional", action="store_true")
    ap.add_argument("--out", default="results/hardware")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    backend = RT.get_backend(args.backend, args.fake, args.fractional)
    name = "fake_fez" if args.fake else backend.name
    stamp = time.strftime("%Y%m%d-%H%M%S")
    record = dict(task=args.task, backend=name, p=args.p, shots=args.shots, started=stamp, results=[])

    if args.task == "reservoir":
        rng = np.random.default_rng(0); T = 40
        s, m = rng.uniform(0, 1, T), rng.uniform(0, 1, T)
        inp = np.column_stack([s, m])
        qcs = [C.reservoir_circuit(inp, tau=t, seed=r) for r, t in enumerate((0.6, 1.0, 1.4, 1.8))]
        isa = RT.to_isa(qcs, backend)
        t0 = time.time(); job = RT.make_sampler(backend, args.shots, twirl=False).run(isa); res = job.result()
        feats = []
        for r in range(len(qcs)):
            bits = RT.counts_to_bits(res[r].data.feat.get_counts(), T * 2)
            z = 1.0 - 2.0 * bits.reshape(len(bits), T, 2)
            feats.append(np.concatenate([z.mean(0), (z[:, :, 0] * z[:, :, 1]).mean(0)[:, None]], axis=1))
        F = np.hstack(feats)
        ideal = np.hstack([RS.CollisionReservoir(3, 2, t, r, trotter=True).run(inp)
                           for r, t in enumerate((0.6, 1.0, 1.4, 1.8))])
        record.update(features=F.tolist(), ideal_features=ideal.tolist(), inputs=inp.tolist(),
                      feature_rmse=float(np.sqrt(np.mean((F - ideal) ** 2))),
                      circuits=[RT.circuit_report(c) for c in isa], wall_s=time.time() - t0,
                      job=RT.job_metadata(job, backend))
        print("feature RMSE vs ideal simulation:", record["feature_rmse"])
    else:
        ang = np.array(json.load(open(args.angles))["angles"][str(args.p)])
        g, b = ang[:args.p], ang[args.p:]
        inst = [load_instance(d) for d in json.load(open(args.instances))[:args.max]]
        n = len(inst[0]["a"])
        if args.task == "estimator":
            est = RT.make_estimator(backend)
            pubs, isas = [], []
            for I in inst:
                qc = C.qaoa_circuit(I["h"], I["K"], args.p, measure=False)
                isa = RT.to_isa(qc, backend); isas.append(isa)
                obs = C.cost_observable(I["h"], I["K"]).apply_layout(isa.layout)
                pubs.append((isa, obs, np.r_[b, g]))        # parameters sort alphabetically: beta, gamma
            t0 = time.time(); job = est.run(pubs); res = job.result()
            for k, I in enumerate(inst):
                sim = NZ.Sim(n); NZ.qaoa_circuit(sim, I["h"], I["K"], g, b)
                z = 1 - 2 * Q.all_bits(n)
                ideal = float(sim.probs() @ (z @ I["h"] + np.einsum("bi,ij,bj->b", z, np.triu(I["K"], 1), z)))
                record["results"].append(dict(E_mitigated=float(res[k].data.evs), E_std=float(res[k].data.stds),
                                              E_ideal=ideal, circuit=RT.circuit_report(isas[k])))
                print(k, record["results"][-1]["E_mitigated"], "ideal", ideal)
        else:
            sampler = RT.make_sampler(backend, args.shots)
            if args.task == "iceberg":
                build = lambda I: C.iceberg_qaoa(I["h"], I["K"], args.p)
                nq = n + 2
            elif args.routing == "swap":
                build = lambda I: C.swap_network_qaoa(I["h"], I["K"], args.p); nq = n
            else:
                build = lambda I: C.qaoa_circuit(I["h"], I["K"], args.p); nq = n
            isas = []
            for I in inst:
                qc = build(I)
                # bind by parameter name so that ordering can never be confused
                bind = {pv: (g[pv.index] if pv.name.startswith("gamma") else b[pv.index]) for pv in qc.parameters}
                isas.append(RT.to_isa(qc.assign_parameters(bind), backend))
            t0 = time.time(); job = sampler.run(isas); res = job.result()
            for k, I in enumerate(inst):
                bits = RT.counts_to_bits(res[k].data.meas.get_counts(), nq)
                r = dict(circuit=RT.circuit_report(isas[k]))
                if args.task == "iceberg":
                    bits, ok = IB.decode(bits); r["accept_fraction"] = float(ok.mean())
                r.update(score(bits, I)); record["results"].append(r)
                print(k, {kk: vv for kk, vv in r.items() if kk not in ("circuit", "mask")})
        record.update(wall_s=time.time() - t0, job=RT.job_metadata(job, backend))
    path = out / f"{args.task}_{name}_{stamp}.json"
    json.dump(record, open(path, "w"), default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
