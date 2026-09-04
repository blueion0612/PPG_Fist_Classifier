"""Render the pipeline figure used at the top of the README.

    python docs/figures/make_hero.py

Writes hero_pipeline.png and hero_pipeline-dark.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THEMES = {
    "light": dict(bg="white", ink="#1c2530", muted="#5b6875", line="#b9c3cf",
                  fill="#eef2f6", a="#4a7fb5", b="#c8683f", o="#3f7d5a",
                  fa="#eaf1f8", fb="#fbeee7", fo="#e9f2ec"),
    "dark":  dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", line="#3d444d",
                  fill="#161b22", a="#6ea8dd", b="#e08a5c", o="#5aa87a",
                  fa="#12202f", fb="#2a1c14", fo="#12241a"),
}


def render(theme, out):
    T = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 4.7), dpi=170)
    ax.set_xlim(0, 92)
    ax.set_ylim(0, 47)
    ax.axis("off")
    fig.patch.set_facecolor(T["bg"])

    def box(x, y, w, h, title, sub, edge=None, face=None, tcol=None):
        edge = edge or T["line"]
        face = face or T["fill"]
        tcol = tcol or T["ink"]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    linewidth=1.4, edgecolor=edge, facecolor=face, zorder=2))
        ax.text(x + w / 2, y + h / 2 + 1.9, title, ha="center", va="center",
                fontsize=11.5, color=tcol, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                fontsize=9.2, color=T["muted"], zorder=3)

    def arrow(x0, y0, x1, y1, c=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.4, color=c or T["line"], shrinkA=0, shrinkB=0, zorder=1))

    W, H, X0 = 25.0, 11.0, 6.0
    R1, R2, R3 = 34.0, 17.0, 3.0
    TRUNK, BUS = 2.4, 30.0
    yA, yB = R2 + H / 2, R3 + H / 2

    box(X0, R1, W, H, "Wrist PPG", "16 channels, 25 Hz")
    arrow(X0 + W, R1 + H / 2, X0 + W + 3, R1 + H / 2)
    box(X0 + W + 3, R1, W, H, "Bandpass 0.5-10 Hz", "4th order, zero phase")
    arrow(X0 + 2 * W + 3, R1 + H / 2, X0 + 2 * W + 6, R1 + H / 2)
    box(X0 + 2 * W + 6, R1, W, H, "Window 3.0 s", "stride 0.5 s, 1.0 s guard")

    SPLIT = X0 + 2 * W + 6 + W / 2
    ax.plot([SPLIT, SPLIT], [R1, BUS], color=T["line"], lw=1.4, zorder=1)
    ax.plot([TRUNK, SPLIT], [BUS, BUS], color=T["line"], lw=1.4, zorder=1)
    ax.plot([TRUNK, TRUNK], [yB, BUS], color=T["line"], lw=1.4, zorder=1)
    arrow(TRUNK, yA, X0, yA, T["a"])
    arrow(TRUNK, yB, X0, yB, T["b"])

    box(X0, R2, W, H, "224 features", "14 per channel", T["a"], T["fa"])
    arrow(X0 + W, yA, X0 + W + 3, yA, T["a"])
    box(X0 + W + 3, R2, W + 2, H, "HistGradientBoosting", "deployed path", T["a"], T["fa"])

    box(X0, R3, W, H, "3 key channels", "ch01, ch05, ch07 raw", T["b"], T["fb"])
    arrow(X0 + W, yB, X0 + W + 3, yB, T["b"])
    box(X0 + W + 3, R3, W + 2, H, "Multi-scale CNN", "reported path", T["b"], T["fb"])

    RE = X0 + 2 * W + 5
    ax.plot([RE, RE + 3], [yA, yA], color=T["a"], lw=1.4, zorder=1)
    ax.plot([RE, RE + 3], [yB, yB], color=T["b"], lw=1.4, zorder=1)
    ax.plot([RE + 3, RE + 3], [yB, yA], color=T["line"], lw=1.4, zorder=1)
    MY = (yA + yB) / 2
    arrow(RE + 3, MY, RE + 6, MY, T["o"])
    box(RE + 6, MY - H / 2, 23.0, H, "open / fist", "per 0.5 s window", T["o"], T["fo"], T["o"])

    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    print("wrote", out)


import os
D = os.path.dirname(os.path.abspath(__file__)) + "/"
render("light", D + "hero_pipeline.png")
render("dark", D + "hero_pipeline-dark.png")
