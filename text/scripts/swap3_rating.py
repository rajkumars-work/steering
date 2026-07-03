"""
Stage A -- swap-3 (star rating as key), authorized 2026-07-03.

Re-keys the SAME 3,600-review pull (distributions/stageA_reviews.parquet,
which already carries the `rating` field -- no re-download needed) by star
rating instead of category or length-bucket, binned into 3 bins:
  low  = 1-2 stars
  mid  = 3 stars
  high = 4-5 stars

Scores the SAME two headline targets already computed in the parquet
(L_sentiment, L_readability) -- no new targets, per the handoff's explicit
"clean apples-to-apples swap-3, not a fishing expedition" instruction.

Reuses the exact TE / bootstrap-CI code from collect_stagea.py (imported,
not re-derived) so the audit convention is identical to the prior six
combinations.

Writes:
  distributions/stageA_perclass_rating.npz   -- shadow, key=ratingbucket
  distributions/stageA_summary_swap3.json    -- T,E,E/(E+T) + bootstrap CIs

Usage: python swap3_rating.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect_stagea import summarize  # reuse existing TE/bootstrap/scoring convention

TARGETS = ["L_sentiment", "L_readability"]


def ratingbucket(r):
    if r <= 2:
        return "low"
    if r == 3:
        return "mid"
    return "high"


def main():
    path = os.path.join(OUT, "stageA_reviews.parquet")
    df = pd.read_parquet(path)
    assert "rating" in df.columns, "rating field missing from saved parquet -- would need a re-pull"

    df["ratingbucket"] = df["rating"].apply(ratingbucket)
    n_per_bin = {b: int((df["ratingbucket"] == b).sum()) for b in ["low", "mid", "high"]}
    print("Per-bin n (rating key, swap-3):", n_per_bin)

    pc, res = summarize(df, "ratingbucket", TARGETS, "ratingbucket")
    np.savez(os.path.join(OUT, "stageA_perclass_rating.npz"), **pc)

    summary = {
        "key": "rating (swap-3)",
        "n_total": int(len(df)),
        "n_per_bin": n_per_bin,
        "conditions": res,
    }
    with open(os.path.join(OUT, "stageA_summary_swap3.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Stage A gate check: key=rating (swap-3, bins=low/mid/high) ===")
    for t in TARGETS:
        s = res[t]
        print(f"  {t:16s} T={s['T']:8.4f} E={s['E']:8.4f} E/(E+T)={s['E_frac']:.3f}  "
              f"CI95={tuple(round(x,3) for x in s['EfracCI95'])}")

    frac_sent = res["L_sentiment"]["E_frac"]
    frac_read = res["L_readability"]["E_frac"]
    ci_sent = res["L_sentiment"]["EfracCI95"]
    ci_read = res["L_readability"]["EfracCI95"]
    sep = (frac_read < ci_sent[0]) or (frac_sent < ci_read[0])  # non-overlapping CIs, either direction
    print(f"\nCIs overlap: {not sep}")
    print("STAGEA-SWAP3-DONE ->", os.path.join(OUT, "stageA_summary_swap3.json"))


if __name__ == "__main__":
    main()
