"""A-priori power analysis (Section 4.2, Eq. 11).

Reproduces the G*Power 3.1 computation for the 2 (system) x 3 (stage) x 2
(region) design: alpha = .05, f = 0.25 (partial eta^2 ~ .06), 1 - beta = .80.

G*Power's "ANOVA: fixed effects, omnibus, one-way" over the 12 design cells
(numerator df = 11) gives N = 279, i.e. 24 participants per cell after rounding
to equal cell sizes -> N = 288 (the value reported in the paper).  With ~5 %
attrition the target was rounded up to N = 300 (25 per cell).

Usage::

    python -m analysis.power_analysis
    python -m analysis.power_analysis --f 0.20 --cells 12 --df1 1
"""

from __future__ import annotations

import argparse
import math

from scipy import stats


def f_from_eta2(eta2_p: float) -> float:
    """Eq. (11): f = sqrt(eta_p^2 / (1 - eta_p^2))."""
    return math.sqrt(eta2_p / (1.0 - eta2_p))


def eta2_from_f(f: float) -> float:
    return f * f / (1.0 + f * f)


def anova_power(n_total: int, f: float, df1: int, n_cells: int, alpha: float = 0.05) -> float:
    """Power of the fixed-effects F test with noncentrality lambda = f^2 * N (G*Power convention)."""
    df2 = n_total - n_cells
    if df2 <= 0:
        return 0.0
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    return float(1 - stats.ncf.cdf(f_crit, df1, df2, f * f * n_total))


def required_n(f: float = 0.25, alpha: float = 0.05, power: float = 0.80, df1: int = 11, n_cells: int = 12,
               equal_cells: bool = True) -> dict:
    n = n_cells + 2
    while anova_power(n, f, df1, n_cells, alpha) < power:
        n += 1
    per_cell = math.ceil(n / n_cells)
    n_equal = per_cell * n_cells if equal_cells else n
    return {"f": f, "eta2_p": eta2_from_f(f), "alpha": alpha, "target_power": power, "df1": df1,
            "n_cells": n_cells, "n_min": n, "n_per_cell": per_cell, "n_total": n_equal,
            "achieved_power": anova_power(n_equal, f, df1, n_cells, alpha)}


def with_attrition(n_total: int, attrition: float = 0.05) -> float:
    """Inflate N for expected attrition (N * (1 + attrition))."""
    return n_total * (1 + attrition)


def sensitivity(n_total: int = 300, alpha: float = 0.05, power: float = 0.80, df1: int = 1, n_cells: int = 12) -> float:
    """Smallest detectable f for a given N (post-hoc sensitivity analysis)."""
    lo, hi = 1e-4, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if anova_power(n_total, mid, df1, n_cells, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--df1", type=int, default=11)
    p.add_argument("--cells", type=int, default=12)
    args = p.parse_args(argv)

    print(f"Eq. (11): eta_p^2 = 0.06 -> f = {f_from_eta2(0.06):.3f}")
    r = required_n(args.f, args.alpha, args.power, args.df1, args.cells)
    print(f"f = {r['f']:.2f} (eta_p^2 = {r['eta2_p']:.3f}), alpha = {r['alpha']}, power = {r['target_power']}, "
          f"df1 = {r['df1']}, cells = {r['n_cells']}")
    print(f"  minimum N = {r['n_min']}  ->  {r['n_per_cell']} per cell  ->  N = {r['n_total']} "
          f"(achieved power {r['achieved_power']:.3f})")
    print(f"  with 5 % attrition: N * 1.05 = {with_attrition(r['n_total']):.1f} "
          f"-> recruited N = 300 (25 per cell) in the study")
    print("\nEffect-specific tests at N = 300 (12 cells):")
    for name, df1 in [("main effect, 2 levels / 2-way with a 2-level factor (df1=1)", 1),
                      ("stage main effect / stage interactions (df1=2)", 2)]:
        print(f"  {name}: power = {anova_power(300, args.f, df1, 12, args.alpha):.3f}, "
              f"min detectable f = {sensitivity(300, args.alpha, args.power, df1):.3f}")


if __name__ == "__main__":
    main()
