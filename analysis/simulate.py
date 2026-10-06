"""SYNTHETIC demo data - NOT the study data.

Generates per-student post-test scores whose cell means and SDs match the
summary statistics reported in Tables 3-5, so the analysis and plotting code
can be exercised end-to-end before the real score sheet is plugged in.
Every row carries ``synthetic = True``; never report results computed on it.
Secondary endpoints (SUS / IPQ / Likert) are left empty because the paper does
not report their values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "data" / "reported_summary_stats.csv"


def _exact_sample(rng, n, mean, sd, lo=0.0, hi=100.0, iters=50):
    x = rng.standard_normal(n)
    for _ in range(iters):
        x = (x - x.mean()) / x.std(ddof=1) * sd + mean
        x = np.clip(x, lo, hi)
        if abs(x.mean() - mean) < 1e-6 and abs(x.std(ddof=1) - sd) < 1e-6:
            break
    return np.round(x, 2)


def simulate(seed: int = 0, summary_path: str | Path = SUMMARY) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = pd.read_csv(summary_path)
    rows = []
    for _, c in cells.iterrows():
        post = _exact_sample(rng, int(c["n"]), c["mean"], c["sd"])
        # matched baseline (Section 4.5: equal numbers of correct answers on a 20-item pretest)
        pre = np.round(np.clip(rng.normal(14, 2.0, int(c["n"])), 0, 20)) * 5
        for k in range(int(c["n"])):
            rows.append({"student_id": f"SYN-{c['stage'][0].upper()}{c['region'][0].upper()}{c['condition'][0].upper()}{k + 1:02d}",
                         "stage": c["stage"], "grade": int(c["grade"]), "region": c["region"],
                         "condition": c["condition"], "pretest": pre[k], "posttest": post[k],
                         "sus": np.nan, "ipq": np.nan, "voice_pref": np.nan, "pacing_match": np.nan,
                         "synthetic": True})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = HERE / "data" / "synthetic_scores_DEMO_ONLY.csv"
    simulate().to_csv(out, index=False)
    print(f"wrote {out} (synthetic, for pipeline testing only)")
