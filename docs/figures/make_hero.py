"""Draw the README figures from results.json.

    python docs/figures/make_hero.py

Writes, each as a light and a -dark variant:

    hero_scenarios.png   F1 and AUC for the six evaluation scenarios, the README hero
    sessions_loso.png    per-session F1 with that session held out
    pipeline.png         the signal chain, drawn, no data behind it

results.json is written by scripts/report_experiment.py. The figures read it
instead of carrying numbers of their own, so they cannot disagree with the
README table, which tests/test_readme_numbers.py checks against the same file.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
sys.path.insert(0, HERE)

import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

with open(HERE + "results.json", encoding="utf-8") as fh:
    R = json.load(fh)

# Display order and label for each scenario key in results.json.
SCENARIOS = [
    ("LOSO (Zero-shot)", "Held-out session\nno calibration"),
    ("Few-shot (10s)", "Calibrated\n10 s"),
    ("Few-shot (20s)", "Calibrated\n20 s"),
    ("Few-shot (30s)", "Calibrated\n30 s"),
    ("Within-Session (User-dep)", "Within one\nsession"),
    ("LOSO + Baseline Sub", "Baseline\nsubtraction"),
]
LEAD = "Few-shot (20s)"  # the result the README leads with


def hero(T):
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 4.5))
    keys = [k for k, _ in SCENARIOS]
    f1 = [R["scenarios"][k]["f1"] for k in keys]
    auc = [R["scenarios"][k]["auc"] for k in keys]
    x = list(range(len(keys)))
    w = 0.36

    ax.yaxis.grid(True, color=T["line"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.bar([i - w / 2 for i in x], f1, w, color=T["green"], label="F1", zorder=3)
    ax.bar([i + w / 2 for i in x], auc, w, color=T["gold"], label="AUC", zorder=3)

    for i, k in enumerate(keys):
        lead = k == LEAD
        for dx, v in ((-w / 2, f1[i]), (w / 2, auc[i])):
            ax.text(i + dx, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=figstyle.SMALL, fontfamily=figstyle.MONO,
                    color=T["ink"] if lead else T["muted"],
                    fontweight="bold" if lead else "normal", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in SCENARIOS])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    figstyle.mono_ticks(ax)
    ax.set_ylabel("score, mean over seven sessions")
    ax.legend(loc="upper left", ncol=2, handlelength=1.1, columnspacing=1.4,
              borderaxespad=0.2)
    return fig


def sessions(T):
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 3.4))
    names = list(R["loso_session_f1"])
    vals = [R["loso_session_f1"][n] for n in names]
    mean = R["loso_mean_f1"]

    ax.yaxis.grid(True, color=T["line"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.bar(names, vals, 0.62, color=[T["green"] if v >= 0.2 else T["muted"] for v in vals],
           zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                fontsize=figstyle.SMALL, fontfamily=figstyle.MONO, color=T["muted"], zorder=4)
    ax.axhline(mean, color=T["gold"], linestyle=(0, (6, 4)), linewidth=1.4, zorder=4)
    # Label the mean above the lowest bar, where nothing else is drawn near the line.
    lo = min(range(len(vals)), key=lambda i: vals[i])
    ax.text(lo, mean + 0.025, f"mean {mean:.3f}", ha="center", va="bottom",
            fontsize=figstyle.SMALL, fontfamily=figstyle.MONO, color=T["gold"], zorder=4)

    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    figstyle.mono_ticks(ax)
    ax.set_ylabel("F1, session held out")
    return fig


def pipeline(T):
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 4.7))
    ax.set_xlim(0, 92)
    ax.set_ylim(0, 47)
    ax.axis("off")

    def box(x, y, w, h, title, sub, edge=None, face=None, tcol=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    linewidth=1.4, edgecolor=edge or T["line"],
                                    facecolor=face or T["fill"], zorder=2))
        ax.text(x + w / 2, y + h / 2 + 1.9, title, ha="center", va="center",
                fontsize=figstyle.TITLE, color=tcol or T["ink"], fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                fontsize=figstyle.SMALL, color=T["muted"], zorder=3)

    def arrow(x0, y0, x1, y1, c=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.4, color=c or T["line"],
                                     shrinkA=0, shrinkB=0, zorder=1))

    def line(xs, ys, c=None):
        ax.plot(xs, ys, color=c or T["line"], linewidth=1.4, zorder=1)

    W, H, X0 = 25.0, 11.0, 6.0
    R1, R2, R3 = 34.0, 17.0, 3.0
    TRUNK, BUS = 2.4, 30.0
    yA, yB = R2 + H / 2, R3 + H / 2
    G, GF, D, DF = T["green"], T["green_fill"], T["gold"], T["gold_fill"]

    box(X0, R1, W, H, "Wrist PPG", "16 channels, 25 Hz")
    arrow(X0 + W, R1 + H / 2, X0 + W + 3, R1 + H / 2)
    box(X0 + W + 3, R1, W, H, "Bandpass 0.5-10 Hz", "4th order, zero phase")
    arrow(X0 + 2 * W + 3, R1 + H / 2, X0 + 2 * W + 6, R1 + H / 2)
    box(X0 + 2 * W + 6, R1, W, H, "Window 3.0 s", "stride 0.5 s, 1.0 s guard")

    split = X0 + 2 * W + 6 + W / 2
    line([split, split], [R1, BUS])
    line([TRUNK, split], [BUS, BUS])
    line([TRUNK, TRUNK], [yB, BUS])
    arrow(TRUNK, yA, X0, yA, G)
    arrow(TRUNK, yB, X0, yB, D)

    box(X0, R2, W, H, "224 features", "14 per channel", G, GF)
    arrow(X0 + W, yA, X0 + W + 3, yA, G)
    box(X0 + W + 3, R2, W + 2, H, "HistGradientBoosting", "deployed path", G, GF)

    box(X0, R3, W, H, "3 key channels", "ch01, ch05, ch07 raw", D, DF)
    arrow(X0 + W, yB, X0 + W + 3, yB, D)
    box(X0 + W + 3, R3, W + 2, H, "Multi-scale CNN", "reported path", D, DF)

    re_ = X0 + 2 * W + 5
    line([re_, re_ + 3], [yA, yA], G)
    line([re_, re_ + 3], [yB, yB], D)
    line([re_ + 3, re_ + 3], [yB, yA])
    my = (yA + yB) / 2
    arrow(re_ + 3, my, re_ + 6, my, G)
    box(re_ + 6, my - H / 2, 23.0, H, "open / fist", "per 0.5 s window", G, GF, G)
    return fig


if __name__ == "__main__":
    figstyle.save_both(hero, HERE + "hero_scenarios")
    figstyle.save_both(sessions, HERE + "sessions_loso")
    figstyle.save_both(pipeline, HERE + "pipeline")
