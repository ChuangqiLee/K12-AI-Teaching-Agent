"""Result figures in the layout of Figures 9-14.

* ``fig_student_scores``  - per-student post-test scores of the four cells (Figures 9, 11, 13).
* ``fig_stage_panel``     - six-panel analysis (Figures 10, 12, 14): box plot, histogram,
                            mean +- SD, individual scores, method x region means, AI gain.

Encoding: hue = instructional condition (blue = AI agent, orange = traditional),
marker / line style = region (circle + solid = rural, square + dashed = urban).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLOR = {"ai_agent": "#2a78d6", "traditional": "#eb6834"}
MARKER = {"rural": "o", "urban": "s"}
LINESTYLE = {"rural": "-", "urban": "--"}
LABEL = {("rural", "traditional"): "Rural Trad", ("urban", "traditional"): "Urban Trad",
         ("rural", "ai_agent"): "Rural AI", ("urban", "ai_agent"): "Urban AI"}
CELLS = [("rural", "traditional"), ("urban", "traditional"), ("rural", "ai_agent"), ("urban", "ai_agent")]
STAGE_TITLE = {"primary": "primary school", "junior": "middle school", "senior": "high school"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.title.set_color(INK)


def _cell(df, stage, region, cond):
    return df[(df.stage == stage) & (df.region == region) & (df.condition == cond)]["posttest"].to_numpy()


def fig_student_scores(df: pd.DataFrame, stage: str, out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for region, cond in CELLS:
        y = _cell(df, stage, region, cond)
        ax.plot(np.arange(1, len(y) + 1), y, color=COLOR[cond], ls=LINESTYLE[region], lw=1.5,
                marker=MARKER[region], ms=6, mfc="white", mew=1.5, label=LABEL[(region, cond)])
    ax.set_xlabel("Students", color=INK2)
    ax.set_ylabel("Scores", color=INK2)
    ax.set_title(f"Score performance of traditional and AI instruction among urban and rural {STAGE_TITLE[stage]} students",
                 fontsize=11)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9)
    _style(ax)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_stage_panel(df: pd.DataFrame, stage: str, out: str | Path) -> Path:
    data = {c: _cell(df, stage, *c) for c in CELLS}
    labels = [LABEL[c] for c in CELLS]
    colors = [COLOR[c[1]] for c in CELLS]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    (a1, a2, a3), (a4, a5, a6) = axes

    bp = a1.boxplot([data[c] for c in CELLS], patch_artist=True, widths=0.5)
    for patch, c, cell in zip(bp["boxes"], colors, CELLS):
        patch.set(facecolor="white", edgecolor=c, linewidth=1.5, hatch="///" if cell[0] == "urban" else "")
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.5)
    a1.set_xticks(range(1, 5), labels, rotation=20)
    a1.set_title("Score distribution (box plot)")
    a1.set_ylabel("Score points", color=INK2)

    lo = min(v.min() for v in data.values())
    bins = np.linspace(np.floor(lo), 100, 9)
    for c, col in zip(CELLS, colors):
        a2.hist(data[c], bins=bins, histtype="step", linewidth=2, color=col, ls=LINESTYLE[c[0]], label=LABEL[c])
    a2.set_title("Score distribution (histogram)")
    a2.set_xlabel("Score points", color=INK2)
    a2.set_ylabel("Number of students", color=INK2)
    a2.legend(frameon=False, fontsize=8)

    means = [data[c].mean() for c in CELLS]
    sds = [data[c].std(ddof=1) for c in CELLS]
    bars = a3.bar(range(4), means, yerr=sds, color=colors, width=0.6, capsize=4, error_kw={"ecolor": INK2},
                  edgecolor="white", linewidth=2)
    for b, cell in zip(bars, CELLS):
        if cell[0] == "urban":
            b.set_hatch("//")
    for i, m in enumerate(means):
        a3.text(i, m + sds[i] * 1.06, f"{m:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    a3.set_xticks(range(4), labels, rotation=20)
    a3.set_ylim(max(0, min(means) - 2.5 * max(sds)), min(105, max(means) + 2 * max(sds)))
    a3.set_title("Average score (mean ± SD)")

    rng = np.random.default_rng(0)
    for i, (c, col) in enumerate(zip(CELLS, colors)):
        a4.scatter(i + rng.uniform(-0.12, 0.12, len(data[c])), data[c], s=36, marker=MARKER[c[0]],
                   facecolor="white", edgecolor=col, linewidth=1.5)
    a4.set_xticks(range(4), labels, rotation=20)
    a4.set_title("Individual student scores")
    a4.set_ylabel("Score points", color=INK2)

    x = np.arange(2)
    for k, cond in enumerate(["traditional", "ai_agent"]):
        vals = [data[("rural", cond)].mean(), data[("urban", cond)].mean()]
        bb = a5.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=COLOR[cond],
                    label="Traditional" if cond == "traditional" else "AI agent", edgecolor="white", linewidth=2)
        for b, v in zip(bb, vals):
            a5.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    a5.set_xticks(x, ["Rural area", "Urban area"])
    a5.set_ylim(max(0, min(means) - 10), min(105, max(means) + 4))
    a5.set_title("Teaching method by region")
    a5.legend(frameon=False, fontsize=8)

    gains = [data[("rural", "ai_agent")].mean() - data[("rural", "traditional")].mean(),
             data[("urban", "ai_agent")].mean() - data[("urban", "traditional")].mean()]
    gb = a6.bar(x, gains, width=0.5, color=COLOR["ai_agent"], edgecolor="white", linewidth=2)
    gb[1].set_hatch("//")
    a6.axhline(0, color=INK2, linewidth=1)
    for b, g in zip(gb, gains):
        a6.text(b.get_x() + b.get_width() / 2, g + (0.04 if g >= 0 else -0.04) * max(abs(x_) for x_ in gains),
                f"{g:+.1f} pts", ha="center", va="bottom" if g >= 0 else "top", fontsize=10, color=INK)
    span = max(abs(g) for g in gains) or 1.0
    a6.set_ylim(min(0, min(gains)) - 0.2 * span, max(0, max(gains)) + 0.2 * span)
    a6.set_xticks(x, ["Rural area", "Urban area"])
    a6.set_title("AI teaching improvement (AI − traditional)")
    a6.set_ylabel("Score difference", color=INK2)

    for ax in axes.flat:
        _style(ax)
    fig.suptitle(f"{STAGE_TITLE[stage].capitalize()} student score analysis: traditional vs AI agent teaching",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
