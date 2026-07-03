"""
Reader-facing verification of the Stage-A (DATA-SIDE) go/no-go gate, from the
tiny per-bin SHADOW alone -- no Amazon Reviews-2023 download, no nltk/textstat,
no model, no GPU.

Inputs (derived statistics only -- category name + per-bin w_b,g_b,v_b for
four targets; no review text, license-clean):
  stageA_perclass.npz            -- key=category,   targets: L_sentiment, L_readability,
                                     L_sentlen, L_firstperson
  stageA_perclass_lenbucket.npz  -- key=length-bucket (swap-2), targets: L_sentiment, L_readability

Reproduces, in well under a second:
  - T, E, T+E and E/(E+T) for every (key, target) combination the Stage-A
    go/no-go gate tried (original pair + both allowed swaps)
  - the gate verdict: did any pair show one target CLEARLY higher E/(E+T)
    than the other, away from the trivial extremes?

This experiment's Stage-A gate FAILED (see stageA_summary.json / RESULTS.md):
every combination lands in E/(E+T) ~ 0.02-0.07, all heavily within each
other's bootstrap CIs -- individual-reviewer variance (T) swamps
product-category variance (E) for every tone/style/complexity proxy tried.
verify_dataside.py exists to let a reader check that claim without re-pulling
the corpus.
"""
import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")


def TE(count, g, v):
    N = count.sum(); w = count / N
    gbar = float((w * g).sum())
    T = float((w * v).sum())
    E = float((w * (g - gbar) ** 2).sum())
    return T, E, gbar


def report(npz_path, targets, key_name):
    d = np.load(npz_path, allow_pickle=True)
    count = d["count"]
    cats = list(d["categories"])
    print(f"--- key = {key_name}  ({int(count.sum())} reviews over {len(cats)} bins: {cats}) ---")
    rows = []
    for t in targets:
        T, E, gbar = TE(count, d[f"g_{t}"], d[f"v_{t}"])
        frac = E / (E + T) if (E + T) > 0 else 0.0
        rows.append((t, T, E, frac))
        print(f"  {t:16s} T={T:10.5g} E={E:10.5g} T+E={T+E:10.5g} E/(E+T)={frac:.3f}")
    return rows


def main():
    print("=== Stage A shadow-only verification (no corpus, no model, no GPU) ===\n")
    rows_cat = report(os.path.join(OUT, "stageA_perclass.npz"),
                       ["L_sentiment", "L_readability", "L_sentlen", "L_firstperson"],
                       "category (original pair + swap-1 alternatives)")
    print()
    rows_lb = report(os.path.join(OUT, "stageA_perclass_lenbucket.npz"),
                      ["L_sentiment", "L_readability"], "length-bucket (swap-2)")

    print("\n=== Gate verdict ===")
    all_fracs = [f for _, _, _, f in rows_cat] + [f for _, _, _, f in rows_lb]
    lo, hi = min(all_fracs), max(all_fracs)
    print(f"E/(E+T) range across every (key,target) combination tried: [{lo:.3f}, {hi:.3f}]")
    print("Go/no-go rule: proceed to Stage B only if one target is CLEARLY higher E/(E+T)")
    print("than the other, both away from the trivial extremes (0 or 1).")
    spread = hi - lo
    verdict = "FAIL (no separation found in any tried combination)" if spread < 0.10 else "borderline -- inspect CIs"
    print(f"Max spread observed: {spread:.3f}  ->  {verdict}")
    print("\nVERIFY-DATASIDE-DONE (shadow only, text domain)")


if __name__ == "__main__":
    main()
