"""Build all manuscript figures from results/*.json  ->  figures/*.png and *.pdf

  python -m experiments.make_figures
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from qespace.drawing import draw

R, F = Path("results"), Path("figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left"})
# Okabe-Ito colour-blind-safe palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
         sky="#56B4E9", grey="#7f7f7f", black="#222222")
W = 7.2   # inches = 183 mm, Nature double-column width


def save(fig, name):
    fig.savefig(F / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def panel(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def sem(x):
    x = np.asarray(x, float); return x.std(ddof=1) / np.sqrt(len(x))


# ------------------------------------------------------------------ Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(W, 3.1)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, title, body, fc, tc=C["black"]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#444", lw=0.7))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=6.3, color="#222", linespacing=1.35)

    def arrow(p, q, **kw):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333", **kw))

    ax.text(11.5, 43, "Biological constraints", ha="center", fontsize=8, fontweight="bold")
    rows = [("Metabolic cost", "ATP pool $A_t$ → budget $B_t$", "#FBE9D0"),
            ("Unreliable release", "Bernoulli $r_{i,t}$, prob. $p_{i,t}$", "#DDE7F3"),
            ("Local credit", "eligibility $e_{i,t}$ × error $m_t$", "#D9EEDD"),
            ("Stable memory", "Fisher $F_i$, anchor $w_i^{\\star}$", "#EDE3F4")]
    for k, (t, b, fc) in enumerate(rows):
        box(1, 32.5 - 10.3 * k, 21, 8.2, t, b, fc)
        arrow((22.6, 36.6 - 10.3 * k), (28.4, 23 + (1.5 - k) * 2.2))
    box(29, 9, 22, 27, "Spiking network",
        "classical recurrent LIF layer\nstochastic synapses\n$n$ dendritic branches\n\ncandidate update $\\delta_k$\nper branch $k$\n\ncost $c_k$, budget $B_t$", "#F2F2F2")
    box(55, 9, 21, 27, "Allocation problem",
        "minimise  $a^{\\top}u+u^{\\top}Ju$\nsubject to  $c^{\\top}u\\leq B_t$\n\n$u_k\\in\\{0,1\\}$: commit\nbranch $k$ or not\n\nIsing $H_C$ on $n$ qubits\n(no slack qubits)", "#FFF6D6")
    box(80, 9, 19, 27, "Quantum processor",
        "QAOA sampler\nsquare lattice /\nheavy-hex\n\nsuppression,\nmitigation,\nerror detection\n\nfeasible best-of-$S$", "#DCEFF7")
    arrow((51.6, 22.5), (54.4, 22.5)); arrow((76.6, 22.5), (79.4, 22.5))
    ax.add_patch(FancyArrowPatch((90, 8.4), (40, 8.4), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color=C["red"]))
    ax.text(65, 0.6, "gated commit  $w \\leftarrow w + \\sum_k u_k\\,\\delta_k$ ;  ATP pool and Fisher anchors updated",
            ha="center", fontsize=6.8, color=C["red"])
    save(fig, "fig1_concept")


# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig = plt.figure(figsize=(W, 3.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.18, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ops = [(0, "h", [q], "H") for q in range(4)] + [(1, "rz", [q], "$R_Z$") for q in range(4)]
    ops += [(2, "rzz", [0, 1], "$R_{ZZ}$"), (2, "rzz", [2, 3], "$R_{ZZ}$"), (3, "rzz", [1, 2], "$R_{ZZ}$"),
            (4, "rzz", [0, 2], "ZZ"), (5, "rzz", [1, 3], "ZZ"), (6, "rzz", [0, 3], "ZZ")]
    ops += [(7, "rx", [q], "$R_X$") for q in range(4)] + [(8, "barrier", [], "")] + [(9, "m", [q], "M") for q in range(4)]
    draw(ax, ["$u_1$", "$u_2$", "$u_3$", "$u_4$"], ops, "a  QAOA layer  $e^{-i\\beta H_M}e^{-i\\gamma H_C}$")
    ax = fig.add_subplot(gs[0, 1])
    ops = [(0, "h", [q], "H") for q in range(5)]
    col = 1
    for layer in range(5):
        for pos in range(layer % 2, 4, 2):
            ops += [(col, "rzz", [pos, pos + 1], "$R_{ZZ}$"), (col + 1, "swap", [pos, pos + 1], "")]
        col += 2
    ops += [(col, "rx", [q], "$R_X$") for q in range(5)]
    draw(ax, [f"$q_{k}$" for k in range(5)], ops, "b  SWAP network (dense couplings, line)")
    ax = fig.add_subplot(gs[1, 0])
    wires = ["t", "$u_1$", "$u_2$", "$u_3$", "$u_4$", "b"]
    ops = [(0, "h", [0], "H")] + [(1 + k, "cx", [k, k + 1], "") for k in range(5)] + [(6, "h", [q], "H") for q in range(6)]
    ops += [(7, "rzz", [1, 5], "ZZ"), (8, "rzz", [1, 2], "$R_{ZZ}$"), (8, "rzz", [3, 4], "$R_{ZZ}$"),
            (9, "rxx", [0, 1], "$R_{XX}$"), (10, "rxx", [0, 2], "XX"), (11, "rxx", [0, 4], "XX"),
            (12, "barrier", [], "")] + [(13, "m", [q], "M") for q in range(6)]
    draw(ax, wires, ops, "c  Error-detected QAOA, [[k+2, k, 2]] code")
    ax.text(6.5, -6.15, "accept shot iff measured parity is even;  $u_i = m_i \\oplus m_b$", fontsize=6.3, ha="center")
    ax = fig.add_subplot(gs[1, 1])
    wires = ["$m_1$", "$m_2$", "$m_3$", "$a_1$", "$a_2$"]
    ops = [(0, "reset", [3], "|0⟩"), (0, "reset", [4], "|0⟩"), (1, "ry", [3], "$R_Y$"), (1, "ry", [4], "$R_Y$"),
           (2.2, "u", [0, 1, 2, 3, 4], "$U(\\tau)$"), (3.4, "m", [3], "M"), (3.4, "m", [4], "M"), (4.4, "barrier", [], ""),
           (5.4, "reset", [3], "|0⟩"), (5.4, "reset", [4], "|0⟩"), (6.4, "ry", [3], "$R_Y$"), (6.4, "ry", [4], "$R_Y$"),
           (7.6, "u", [0, 1, 2, 3, 4], "$U(\\tau)$"), (8.8, "m", [3], "M"), (8.8, "m", [4], "M")]
    draw(ax, wires, ops, "d  Collision-model reservoir, two steps")
    ax.text(1, -4.95, "$\\pi s_t,\\ \\pi m_t$", fontsize=6.3, ha="center")
    ax.text(6.4, -4.95, "$\\pi s_{t+1},\\ \\pi m_{t+1}$", fontsize=6.3, ha="center")
    save(fig, "fig2_circuits")


# ------------------------------------------------------------------ Fig. 3
def fig3():
    d = json.load(open(R / "exp1_continual.json")); S = d["seeds"]
    names = ["dense", "dense_ewc", "random", "uniform", "diag", "greedy", "anneal", "exact", "qaoa"]
    lab = ["Dense", "Dense+EWC", "Random mask", "Uniform sampler", "Diagonal knapsack", "Greedy", "Annealing", "Exact", "QAOA (sim.)"]
    col = [C["grey"], C["grey"], C["sky"], C["sky"], C["orange"], C["orange"], C["orange"], C["black"], C["blue"]]
    g = lambda n, k: np.array([s[n][k] for s in S], float)
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.9), gridspec_kw=dict(hspace=0.62, wspace=0.36))
    rng = np.random.default_rng(0)
    for ax, key, yl, t in ((axs[0, 0], "final_acc", "Mean accuracy after 5 tasks", "Retained accuracy"),
                           (axs[0, 1], "forgetting", "Forgetting (accuracy lost)", "Forgetting")):
        for i, n in enumerate(names):
            v = g(n, key)
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=5, color=col[i], alpha=0.35, lw=0)
            ax.errorbar(i, v.mean(), yerr=sem(v), fmt="o", ms=4, color=col[i], capsize=2, lw=1, mec="white", mew=0.5)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(lab, fontsize=6, rotation=35, ha="right"); ax.set_ylabel(yl); ax.set_title(t)
    panel(axs[0, 0], "a"); panel(axs[0, 1], "b")
    ax = axs[1, 0]
    for i, n in enumerate(names):
        ax.errorbar(g(n, "update_cost").mean(), g(n, "final_acc").mean(), yerr=sem(g(n, "final_acc")), fmt="o", ms=4.5,
                    color=col[i], capsize=2, lw=1, mec="white", mew=0.5)
    for n, t, off in (("dense", "Dense", (-8, -11)), ("dense_ewc", "Dense+EWC", (-10, 6)), ("exact", "budgeted\nallocators", (8, -4)),
                      ("random", "Random mask", (8, -8))):
        ax.annotate(t, (g(n, "update_cost").mean(), g(n, "final_acc").mean()), xytext=off, textcoords="offset points",
                    fontsize=6, ha="left" if off[0] > 0 else "right")
    ax.set_xlabel("Total plasticity cost (proxy units)"); ax.set_ylabel("Mean accuracy after 5 tasks")
    ax.set_title("Accuracy versus plasticity cost"); ax.set_xlim(0, 8200); panel(ax, "c")
    ax = axs[1, 1]
    sel = ["random", "uniform", "diag", "greedy", "anneal", "qaoa"]; x = np.arange(len(sel))
    hit = [g(n, "hit_opt") for n in sel]
    ax.bar(x, [h.mean() for h in hit], yerr=[sem(h) for h in hit], color=[col[names.index(n)] for n in sel], width=0.62,
           error_kw=dict(lw=0.8, capsize=2))
    for i, n in enumerate(sel):
        ax.text(i, hit[i].mean() + 0.035, f"{100 * g(n, 'mean_regret').mean():.1f}%", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([lab[names.index(n)] for n in sel], fontsize=6, rotation=35, ha="right")
    ax.set_ylabel("Fraction of windows solved optimally"); ax.set_ylim(0, 1.08)
    ax.set_title("Allocation quality (label: regret)"); panel(ax, "d")
    save(fig, "fig3_continual")


# ------------------------------------------------------------------ Fig. 4
def fig4():
    d = json.load(open(R / "exp2_qubo.json")); sizes = [str(n) for n in d["sizes"] if str(n) in d["qaoa"]]
    cols = [C["blue"], C["green"], C["red"]]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw=dict(wspace=0.42))
    ax = axs[0]
    for n, c in zip(sizes, cols):
        ps = sorted(int(p) for p in d["qaoa"][n]); m = [np.mean(d["qaoa"][n][str(p)]["p_opt"]) for p in ps]
        e = [sem(d["qaoa"][n][str(p)]["p_opt"]) for p in ps]
        ax.errorbar(ps, m, yerr=e, marker="o", ms=3.5, color=c, lw=1, capsize=2, label=f"n = {n}")
    ax.set_yscale("log"); ax.set_xlabel("QAOA depth p (0 = uniform)"); ax.set_ylabel("Probability of optimal feasible mask")
    ax.set_xticks(ps); ax.legend(fontsize=6, loc="lower right"); ax.set_title("Success probability"); panel(ax, "a")
    ax = axs[1]
    for n, c in zip(sizes, cols):
        ps = sorted(int(p) for p in d["qaoa"][n]); m = [100 * np.mean(d["qaoa"][n][str(p)]["regret"]) for p in ps]
        ax.plot(ps, m, marker="o", ms=3.5, color=c, lw=1)
        cl = d["classical"][n]
        ax.plot([4.25], [100 * cl["greedy_regret"]], marker="s", ms=3.5, color=c, mfc="white")
        ax.plot([4.5], [100 * cl["anneal_regret"]], marker="^", ms=3.5, color=c, mfc="white")
    ax.set_yscale("log"); ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Regret of best of 256 shots (%)"); ax.set_title("Solution quality")
    ax.plot([], [], "s", color="k", mfc="white", ms=3.5, label="greedy"); ax.plot([], [], "^", color="k", mfc="white", ms=3.5, label="annealing")
    ax.legend(fontsize=6, loc="lower left"); panel(ax, "b")
    ax = axs[2]
    for n, c in zip(sizes, cols):
        ps = sorted(int(p) for p in d["qaoa"][n])
        ax.plot(ps, [np.median(d["qaoa"][n][str(p)]["shots99"]) for p in ps], marker="o", ms=3.5, color=c, lw=1)
    ax.set_yscale("log"); ax.set_xticks(ps); ax.set_xlabel("QAOA depth p"); ax.set_ylabel("Shots for 99% success (median)")
    ax.axhline(256, color=C["grey"], lw=0.7, ls="--"); ax.text(4.1, 300, "256 shots", fontsize=6, color=C["grey"], ha="right")
    ax.set_title("Sampling cost"); panel(ax, "c")
    save(fig, "fig4_qubo")


# ------------------------------------------------------------------ Fig. 5
def fig5():
    d = json.load(open(R / "exp3_noise.json"))
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw=dict(wspace=0.42))
    ax = axs[0]; p2 = [b["p2"] for b in d["p2"]]
    keys = [("E_raw", "raw", C["grey"]), ("E_ro", "read-out mitigated", C["orange"]), ("E_zne_lin", "+ ZNE linear", C["sky"]),
            ("E_zne_exp", "+ ZNE exponential", C["blue"])]
    wd = 0.2
    for k, (key, lab, c) in enumerate(keys):
        v = [[abs(r[key] - r["E_ideal"]) / abs(r["E_ideal"]) for r in b["rows"]] for b in d["p2"]]
        ax.bar(np.arange(len(p2)) + (k - 1.5) * wd, [100 * np.mean(x) for x in v], wd, yerr=[100 * sem(x) for x in v], color=c,
               label=lab, error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_xticks(range(len(p2))); ax.set_xticklabels([f"{1e3 * p:g}" for p in p2])
    ax.set_xlabel("Two-qubit error per gate (×10$^{-3}$)"); ax.set_ylabel("Relative error of ⟨$H_C$⟩ (%)")
    ax.set_ylim(0, 33); ax.legend(fontsize=5.6, loc="upper left"); ax.set_title("Expectation values"); panel(ax, "a")
    ax = axs[1]
    idl = np.mean([r["popt_ideal"] for r in d["p2"][0]["rows"]])
    ax.axhline(idl, color=C["black"], lw=0.8, ls="--"); ax.text(len(p2) - 0.5, idl * 1.04, "noiseless", fontsize=6, ha="right")
    v = [[r["popt_noisy"] for r in b["rows"]] for b in d["p2"]]
    ax.bar(range(len(p2)), [np.mean(x) for x in v], 0.55, yerr=[sem(x) for x in v], color=C["red"], error_kw=dict(lw=0.7, capsize=2))
    for i, b in enumerate(d["p2"]):
        ax.text(i, 0.002, f"regret\n{100 * np.mean([r['regret256_noisy'] for r in b['rows']]):.1f}%", ha="center", fontsize=5.8, color="white")
    ax.set_xticks(range(len(p2))); ax.set_xticklabels([f"{1e3 * p:g}" for p in p2])
    ax.set_xlabel("Two-qubit error per gate (×10$^{-3}$)"); ax.set_ylabel("Probability of optimal mask per shot")
    ax.set_title("Samples under noise"); panel(ax, "b")
    ax = axs[2]; ic = d["iceberg"]; x = [1e3 * b["p2"] for b in ic]
    ax.plot(x, [b["mean"]["popt_unenc"] for b in ic], marker="o", ms=3.5, color=C["red"], lw=1, label="unencoded")
    ax.plot(x, [b["mean"]["popt_enc"] for b in ic], marker="s", ms=3.5, color=C["green"], lw=1, label="error-detected (accepted)")
    ax.axhline(ic[0]["mean"]["popt_ideal"], color=C["black"], lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("Two-qubit error per gate (×10$^{-3}$)"); ax.set_ylabel("Probability of optimal mask")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(x, [b["mean"]["accept"] for b in ic], marker="^", ms=3, color=C["grey"], lw=0.8, ls=":")
    ax2.set_ylabel("Accepted fraction", color=C["grey"]); ax2.set_ylim(0, 1.05); ax2.tick_params(colors=C["grey"])
    ax.legend(fontsize=5.6, loc="lower left"); ax.set_title("Error detection"); panel(ax, "c")
    save(fig, "fig5_noise")


# ------------------------------------------------------------------ Fig. 6
def fig6():
    d = json.load(open(R / "exp4_reservoir.json")); S = d["seeds"]
    st = {"quantum": ("Quantum reservoir (exact)", C["blue"], "o", "-"), "quantum_shots": ("Quantum, 2,048 shots", C["sky"], "o", "--"),
          "esn": ("Echo-state network", C["orange"], "s", "-"), "delay": ("Delay line", C["grey"], "^", "-")}
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw=dict(wspace=0.42, width_ratios=[1.25, 1, 1]))
    ax = axs[0]
    for k, (lab, c, m, ls) in st.items():
        r2 = np.array([[x[1] for x in s["memory"][k]] for s in S])
        ax.errorbar(d["taus_mem"], r2.mean(0), yerr=r2.std(0, ddof=1) / np.sqrt(len(S)), marker=m, ms=3.2, color=c, lw=1, ls=ls,
                    capsize=1.5, label=f"{lab} (MC = {r2.sum(1).mean():.1f})")
    ax.set_xlabel("Delay τ (time steps)"); ax.set_ylabel("Recall $r^2$ of $s_{t-\\tau}$"); ax.legend(fontsize=5.4, loc="upper right")
    ax.set_ylim(0, 1.85); ax.set_yticks([0, 0.5, 1]); ax.set_title("Linear memory"); panel(ax, "a")
    for ax, task, t, let in ((axs[1], "product", "Delayed product $s_t s_{t-\\tau}$", "b"), (axs[2], "credit", "Three-factor credit signal", "c")):
        ks = ["quantum", "quantum_shots", "quantum_dephased", "esn", "delay", "oracle"]
        lb = ["Quantum", "Quantum, shots", "Quantum, dephased", "ESN", "Delay line", "Oracle"]
        cs = [C["blue"], C["sky"], C["purple"], C["orange"], C["grey"], C["black"]]
        v = [np.array([np.mean([x[0] for x in s[task][k]]) for s in S]) for k in ks]
        ax.bar(range(len(ks)), [x.mean() for x in v], 0.62, yerr=[sem(x) for x in v], color=cs, error_kw=dict(lw=0.7, capsize=2))
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(lb, fontsize=6, rotation=35, ha="right"); ax.set_ylabel("Test NMSE (lower is better)")
        ax.set_title(t); panel(ax, let)
    save(fig, "fig6_reservoir")


if __name__ == "__main__":
    F.mkdir(exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
