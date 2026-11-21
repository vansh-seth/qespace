"""Minimal matplotlib quantum-circuit drawer (no Qiskit needed) for the figures."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

INK, WIRE = "#1a1a1a", "#555555"
FILL = {"h": "#DDE7F3", "rz": "#FBE9D0", "rzz": "#F6D6CC", "rx": "#D9EEDD", "ry": "#EDE3F4",
        "m": "#EEEEEE", "reset": "#EEEEEE", "u": "#E5E5E5", "rxx": "#D9EEDD"}


def draw(ax, wires, ops, title=None, dy=1.0, fs=7.5):
    """wires: list of labels (top to bottom). ops: list of (column, kind, qubits, text)."""
    n = len(wires); ncol = max(o[0] for o in ops) + 1
    y = lambda q: -(q * dy)
    for q, lab in enumerate(wires):
        ax.plot([-0.6, ncol - 0.4], [y(q), y(q)], color=WIRE, lw=0.8, zorder=1)
        ax.text(-0.75, y(q), lab, ha="right", va="center", fontsize=fs)
    for col, kind, qs, text in ops:
        if kind == "barrier":
            ax.plot([col, col], [y(0) + 0.45, y(n - 1) - 0.45], color="#999999", lw=0.8, ls=(0, (2, 2)), zorder=1)
            continue
        if kind in ("cx", "swap"):
            a, b = qs
            ax.plot([col, col], [y(a), y(b)], color=INK, lw=0.9, zorder=2)
            if kind == "cx":
                ax.add_patch(Circle((col, y(a)), 0.07, color=INK, zorder=3))
                ax.add_patch(Circle((col, y(b)), 0.17, fc="white", ec=INK, lw=0.9, zorder=3))
                ax.plot([col - 0.17, col + 0.17], [y(b), y(b)], color=INK, lw=0.9, zorder=4)
                ax.plot([col, col], [y(b) - 0.17, y(b) + 0.17], color=INK, lw=0.9, zorder=4)
            else:
                for q in qs:
                    ax.plot([col - 0.12, col + 0.12], [y(q) - 0.12, y(q) + 0.12], color=INK, lw=1.1, zorder=3)
                    ax.plot([col - 0.12, col + 0.12], [y(q) + 0.12, y(q) - 0.12], color=INK, lw=1.1, zorder=3)
            continue
        top, bot = min(qs), max(qs)
        if len(qs) == 2 and bot - top > 1:      # non-adjacent two-qubit gate: two boxes + link
            ax.plot([col, col], [y(top), y(bot)], color=INK, lw=0.9, zorder=2)
            for q in qs:
                ax.add_patch(FancyBboxPatch((col - 0.36, y(q) - 0.3), 0.72, 0.6, boxstyle="round,pad=0.02",
                                            fc=FILL.get(kind, "#EEE"), ec=INK, lw=0.7, zorder=3))
            ax.text(col, y(top), text, ha="center", va="center", fontsize=fs - 1, zorder=4)
            ax.text(col, y(bot), text, ha="center", va="center", fontsize=fs - 1, zorder=4)
            continue
        h = (bot - top) * dy + 0.6
        ax.add_patch(FancyBboxPatch((col - 0.36, y(bot) - 0.3), 0.72, h, boxstyle="round,pad=0.02",
                                    fc=FILL.get(kind, "#EEE"), ec=INK, lw=0.7, zorder=3))
        ax.text(col, (y(top) + y(bot)) / 2, text, ha="center", va="center", fontsize=fs - (1 if len(text) > 4 else 0),
                zorder=4, rotation=90 if (len(qs) > 1 and len(text) > 5) else 0)
    ax.set_xlim(-1.9, ncol - 0.2); ax.set_ylim(y(n - 1) - 0.6, y(0) + 0.75)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fs + 1, loc="left", pad=2)
