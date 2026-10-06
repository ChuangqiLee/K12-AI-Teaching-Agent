"""Statistical analysis plan (Sections 4.1, 4.2, 4.6, 4.7).

* Descriptives per cell (Tables 3-5): mean, SD, variance.
* Pooled condition means, AI-minus-traditional gains per region, urban-rural gaps,
  Cohen's d with the pooled SD (as reported in Section 4.6).
* Three-way ANOVA  condition x stage x region  on post-test scores, ANCOVA with the
  pre-test as covariate, planned contrasts (AI vs. traditional within each stage),
  partial eta^2, Levene tests for dispersion differences.
* Secondary endpoints (SUS, IPQ, Likert personalisation fit) when present.

Input format: one row per student, see ``analysis/templates/scores_template.csv``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

STAGES = ["primary", "junior", "senior"]
REGIONS = ["rural", "urban"]
CONDITIONS = ["traditional", "ai_agent"]


# --------------------------------------------------------------------------- summary-level
def combine_groups(ns: Iterable[float], means: Iterable[float], sds: Iterable[float]) -> Dict[str, float]:
    """Exact mean / SD of the union of several groups given their n, mean and SD."""
    ns, means, sds = map(lambda x: np.asarray(list(x), dtype=float), (ns, means, sds))
    n = ns.sum()
    m = float((ns * means).sum() / n)
    ss = float(((ns - 1) * sds ** 2).sum() + (ns * (means - m) ** 2).sum())
    return {"n": float(n), "mean": m, "sd": math.sqrt(ss / (n - 1))}


def cohens_d(m1: float, s1: float, n1: float, m2: float, s2: float, n2: float) -> float:
    """d = (m1 - m2) / pooled SD."""
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (m1 - m2) / sp


def hedges_g(m1, s1, n1, m2, s2, n2) -> float:
    return cohens_d(m1, s1, n1, m2, s2, n2) * (1 - 3 / (4 * (n1 + n2) - 9))


def stage_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """cells: columns stage, region, condition, n, mean, sd -> one row per stage with the paper's indicators."""
    rows = []
    for stage in [s for s in STAGES if s in set(cells["stage"])]:
        c = cells[cells["stage"] == stage].set_index(["region", "condition"])
        get = lambda r, k: c.loc[(r, k)]  # noqa: E731
        trad = combine_groups(*(c.xs("traditional", level="condition")[k] for k in ("n", "mean", "sd")))
        ai = combine_groups(*(c.xs("ai_agent", level="condition")[k] for k in ("n", "mean", "sd")))
        urban = combine_groups(*(c.xs("urban", level="region")[k] for k in ("n", "mean", "sd")))
        rural = combine_groups(*(c.xs("rural", level="region")[k] for k in ("n", "mean", "sd")))
        d = cohens_d(ai["mean"], ai["sd"], ai["n"], trad["mean"], trad["sd"], trad["n"])
        sp = math.sqrt(((ai["n"] - 1) * ai["sd"] ** 2 + (trad["n"] - 1) * trad["sd"] ** 2) / (ai["n"] + trad["n"] - 2))
        rows.append({
            "stage": stage,
            "trad_mean": trad["mean"], "trad_sd": trad["sd"],
            "ai_mean": ai["mean"], "ai_sd": ai["sd"],
            "ai_minus_trad": ai["mean"] - trad["mean"],
            "pooled_sd": sp, "cohens_d": d,
            "gain_urban": get("urban", "ai_agent")["mean"] - get("urban", "traditional")["mean"],
            "gain_rural": get("rural", "ai_agent")["mean"] - get("rural", "traditional")["mean"],
            "gap_trad_abs": abs(get("urban", "traditional")["mean"] - get("rural", "traditional")["mean"]),
            "gap_ai_abs": abs(get("urban", "ai_agent")["mean"] - get("rural", "ai_agent")["mean"]),
            "urban_mean": urban["mean"], "rural_mean": rural["mean"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- student-level
def describe_cells(df: pd.DataFrame, dv: str = "posttest") -> pd.DataFrame:
    g = df.groupby(["stage", "region", "condition"])[dv]
    out = g.agg(n="count", mean="mean", sd="std", var="var", min="min", max="max").reset_index()
    out["stage"] = pd.Categorical(out["stage"], STAGES, ordered=True)
    return out.sort_values(["stage", "condition", "region"]).reset_index(drop=True)


def _partial_eta2(table: pd.DataFrame) -> pd.DataFrame:
    resid = table.loc["Residual", "sum_sq"]
    table = table.copy()
    table["eta2_p"] = table["sum_sq"] / (table["sum_sq"] + resid)
    table.loc["Residual", "eta2_p"] = np.nan
    return table


def three_way_anova(df: pd.DataFrame, dv: str = "posttest", typ: int = 3) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    formula = f"{dv} ~ C(condition, Sum) * C(stage, Sum) * C(region, Sum)"
    model = smf.ols(formula, data=df).fit()
    return _partial_eta2(anova_lm(model, typ=typ))


def ancova(df: pd.DataFrame, dv: str = "posttest", covariate: str = "pretest", typ: int = 3) -> Optional[pd.DataFrame]:
    if covariate not in df or df[covariate].isna().all():
        return None
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    formula = f"{dv} ~ {covariate} + C(condition, Sum) * C(stage, Sum) * C(region, Sum)"
    return _partial_eta2(anova_lm(smf.ols(formula, data=df.dropna(subset=[covariate])).fit(), typ=typ))


def planned_contrasts(df: pd.DataFrame, dv: str = "posttest") -> pd.DataFrame:
    """AI vs. traditional within each stage (and within each stage x region), Welch t, d, Holm-adjusted p."""
    rows = []
    for stage in STAGES:
        for region in [None] + REGIONS:
            sub = df[(df["stage"] == stage) & ((df["region"] == region) if region else True)]
            a = sub.loc[sub["condition"] == "ai_agent", dv].dropna()
            t_ = sub.loc[sub["condition"] == "traditional", dv].dropna()
            if len(a) < 2 or len(t_) < 2:
                continue
            res = stats.ttest_ind(a, t_, equal_var=False)
            lev = stats.levene(a, t_, center="median")
            rows.append({"stage": stage, "region": region or "all", "n_ai": len(a), "n_trad": len(t_),
                         "mean_ai": a.mean(), "mean_trad": t_.mean(), "diff": a.mean() - t_.mean(),
                         "t": res.statistic, "df_welch": _welch_df(a, t_), "p": res.pvalue,
                         "cohens_d": cohens_d(a.mean(), a.std(), len(a), t_.mean(), t_.std(), len(t_)),
                         "sd_ai": a.std(), "sd_trad": t_.std(), "levene_W": lev.statistic, "levene_p": lev.pvalue})
    out = pd.DataFrame(rows)
    if not out.empty:
        from statsmodels.stats.multitest import multipletests
        mask = out["region"] == "all"
        out.loc[mask, "p_holm"] = multipletests(out.loc[mask, "p"], method="holm")[1]
    return out


def _welch_df(a, b) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))


def equity_gaps(df: pd.DataFrame, dv: str = "posttest") -> pd.DataFrame:
    """H4: urban-rural gap per condition and its change, with a condition x region interaction test per stage."""
    import statsmodels.formula.api as smf
    rows = []
    for stage in STAGES:
        sub = df[df["stage"] == stage]
        if sub.empty:
            continue
        m = sub.groupby(["condition", "region"])[dv].mean()
        fit = smf.ols(f"{dv} ~ C(condition) * C(region)", data=sub).fit()
        term = [k for k in fit.pvalues.index if ":" in k][0]
        rows.append({"stage": stage,
                     "gap_trad": m[("traditional", "urban")] - m[("traditional", "rural")],
                     "gap_ai": m[("ai_agent", "urban")] - m[("ai_agent", "rural")],
                     "interaction_b": fit.params[term], "interaction_p": fit.pvalues[term]})
    return pd.DataFrame(rows)


def secondary_endpoints(df: pd.DataFrame, columns: List[str] = ("sus", "ipq", "voice_pref", "pacing_match")) -> pd.DataFrame:
    """H2 / H3: compare UX ratings between conditions per stage (Welch t, d)."""
    rows = []
    for col in columns:
        if col not in df or df[col].isna().all():
            continue
        for stage in STAGES:
            sub = df[df["stage"] == stage]
            a = sub.loc[sub["condition"] == "ai_agent", col].dropna()
            t_ = sub.loc[sub["condition"] == "traditional", col].dropna()
            if len(a) < 2 or len(t_) < 2:
                continue
            res = stats.ttest_ind(a, t_, equal_var=False)
            rows.append({"measure": col, "stage": stage, "mean_ai": a.mean(), "mean_trad": t_.mean(),
                         "t": res.statistic, "p": res.pvalue,
                         "cohens_d": cohens_d(a.mean(), a.std(), len(a), t_.mean(), t_.std(), len(t_))})
    return pd.DataFrame(rows)


def sus_score(responses: Iterable[int]) -> float:
    """System Usability Scale (10 items, 1-5) -> 0-100."""
    r = list(responses)
    if len(r) != 10:
        raise ValueError("SUS has 10 items")
    odd = sum(x - 1 for x in r[0::2])
    even = sum(5 - x for x in r[1::2])
    return (odd + even) * 2.5
