"""Run the complete analysis plan.

    # 1) verify the numbers reported in Section 4.6 from Tables 3-5 (no raw data needed)
    python -m analysis.run_analysis --summary

    # 2) full analysis on the real per-student score sheet
    python -m analysis.run_analysis --data path/to/scores.csv --out results/

    # 3) pipeline test on SYNTHETIC data (never report these numbers)
    python -m analysis.run_analysis --synthetic --out results_synthetic/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import figures, stats
from .power_analysis import required_n
from .simulate import SUMMARY, simulate

pd.set_option("display.width", 160)
pd.set_option("display.precision", 3)


def run_summary() -> pd.DataFrame:
    cells = pd.read_csv(SUMMARY)
    s = stats.stage_summary(cells)
    print("== Section 4.6 indicators recomputed from the reported cell statistics (Tables 3-5) ==")
    print(s.round(3).to_string(index=False))
    return s


def run_full(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    synthetic = bool(df.get("synthetic", pd.Series([False])).any())
    if synthetic:
        print("!! WARNING: synthetic demo data - results are NOT study results !!\n")
    r = required_n()
    print(f"A-priori power: N = {r['n_total']} ({r['n_per_cell']} per cell), observed N = {len(df)}\n")

    desc = stats.describe_cells(df)
    print("== Descriptives (Tables 3-5) ==\n", desc.to_string(index=False), "\n")
    desc.to_csv(out / "descriptives.csv", index=False)

    cells = desc.rename(columns={"n": "n"})[["stage", "region", "condition", "n", "mean", "sd"]]
    summ = stats.stage_summary(cells.astype({"stage": str}))
    print("== Stage summary (gains, gaps, Cohen's d) ==\n", summ.round(3).to_string(index=False), "\n")
    summ.to_csv(out / "stage_summary.csv", index=False)

    anova = stats.three_way_anova(df)
    print("== Three-way ANOVA (type III, sum-to-zero contrasts) ==\n", anova.round(4).to_string(), "\n")
    anova.to_csv(out / "anova_3way.csv")

    anc = stats.ancova(df)
    if anc is not None:
        print("== ANCOVA (pre-test covariate) ==\n", anc.round(4).to_string(), "\n")
        anc.to_csv(out / "ancova.csv")

    con = stats.planned_contrasts(df)
    print("== Planned contrasts: AI vs traditional ==\n", con.round(4).to_string(index=False), "\n")
    con.to_csv(out / "planned_contrasts.csv", index=False)

    eq = stats.equity_gaps(df)
    print("== Equity (H4): urban-rural gaps ==\n", eq.round(4).to_string(index=False), "\n")
    eq.to_csv(out / "equity_gaps.csv", index=False)

    sec = stats.secondary_endpoints(df)
    if not sec.empty:
        print("== Secondary endpoints (H2/H3) ==\n", sec.round(4).to_string(index=False), "\n")
        sec.to_csv(out / "secondary_endpoints.csv", index=False)

    fig_ids = {"primary": (9, 10), "junior": (11, 12), "senior": (13, 14)}
    for stage, (a, b) in fig_ids.items():
        if (df.stage == stage).any():
            figures.fig_student_scores(df, stage, out / f"figure{a}_{stage}_scores.png")
            figures.fig_stage_panel(df, stage, out / f"figure{b}_{stage}_panel.png")
    print(f"tables and figures written to {out}/")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary", action="store_true", help="recompute Section 4.6 indicators from Tables 3-5")
    g.add_argument("--data", type=Path, help="per-student CSV (see analysis/templates/scores_template.csv)")
    g.add_argument("--synthetic", action="store_true", help="run on synthetic demo data")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.summary:
        run_summary()
    elif args.synthetic:
        run_full(simulate(args.seed), args.out)
    else:
        run_full(pd.read_csv(args.data), args.out)


if __name__ == "__main__":
    main()
