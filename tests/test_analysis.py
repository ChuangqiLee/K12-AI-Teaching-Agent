"""The analysis code must reproduce every number reported in Sections 4.2 and 4.6."""

import pandas as pd
import pytest

from analysis import stats
from analysis.power_analysis import f_from_eta2, required_n
from analysis.simulate import SUMMARY, simulate


def test_power_analysis_reproduces_288():
    r = required_n(f=0.25, alpha=0.05, power=0.80, df1=11, n_cells=12)
    assert r["n_total"] == 288 and r["n_per_cell"] == 24
    assert f_from_eta2(0.06) == pytest.approx(0.25, abs=0.005)


@pytest.fixture(scope="module")
def summary():
    return stats.stage_summary(pd.read_csv(SUMMARY)).set_index("stage")


def test_primary_numbers(summary):
    p = summary.loc["primary"]
    assert p.ai_mean == pytest.approx(96.20, abs=0.01) and p.trad_mean == pytest.approx(93.64, abs=0.01)
    assert p.cohens_d == pytest.approx(0.87, abs=0.01)
    assert p.gain_urban == pytest.approx(2.9, abs=0.05) and p.gain_rural == pytest.approx(2.2, abs=0.05)
    assert p.gap_trad_abs == pytest.approx(1.55) and p.gap_ai_abs == pytest.approx(0.85)
    assert p.trad_sd == pytest.approx(3.5, abs=0.06) and p.ai_sd == pytest.approx(2.3, abs=0.02)


def test_junior_numbers(summary):
    j = summary.loc["junior"]
    assert j.trad_mean == pytest.approx(85.58, abs=0.01) and j.ai_mean == pytest.approx(89.68, abs=0.01)
    assert j.ai_minus_trad == pytest.approx(4.09, abs=0.01) and j.cohens_d == pytest.approx(0.66, abs=0.01)
    assert j.urban_mean == pytest.approx(87.98, abs=0.01) and j.rural_mean == pytest.approx(87.28, abs=0.01)
    assert j.gain_rural == pytest.approx(5.16) and j.gain_urban == pytest.approx(3.02)


def test_senior_numbers(summary):
    s = summary.loc["senior"]
    assert s.ai_mean == pytest.approx(65.03, abs=0.01) and s.trad_mean == pytest.approx(74.46, abs=0.01)
    assert s.ai_minus_trad == pytest.approx(-9.43, abs=0.01)
    assert s.pooled_sd == pytest.approx(12.9, abs=0.05) and s.cohens_d == pytest.approx(-0.73, abs=0.01)


def test_full_pipeline_on_synthetic(tmp_path):
    df = simulate(seed=1)
    assert df["synthetic"].all() and len(df) == 300
    desc = stats.describe_cells(df)
    ref = pd.read_csv(SUMMARY)
    merged = desc.astype({"stage": str}).merge(ref, on=["stage", "region", "condition"], suffixes=("", "_ref"))
    assert (merged["mean"] - merged["mean_ref"]).abs().max() < 1e-3
    assert (merged["sd"] - merged["sd_ref"]).abs().max() < 1e-3
    anova = stats.three_way_anova(df)
    assert "eta2_p" in anova and anova.loc["C(condition, Sum):C(stage, Sum)", "PR(>F)"] < 0.001
    con = stats.planned_contrasts(df)
    assert set(con.stage) == {"primary", "junior", "senior"}
    from analysis.run_analysis import run_full
    run_full(df, tmp_path)
    assert (tmp_path / "figure10_primary_panel.png").exists()


def test_sus_score():
    assert stats.sus_score([5, 1] * 5) == 100.0
    assert stats.sus_score([3] * 10) == 50.0
