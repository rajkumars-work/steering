#!/usr/bin/env python3
"""Verify C-E18's own claims: (A) data-side E/(E+T) orders properties the same as the realized C-14
ratio; (B) key_quality is computed correctly and the recipe picks the argmax. Pure CPU; no GPU.
Run: python verify_ce18.py   (after ce18_key_selection.py has written ce18_key_selection.json)."""
import os, json, sys

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "ce18_key_selection.json")))
ok = True
def chk(name, cond):
    global ok; print(("PASS" if cond else "FAIL"), name); ok = ok and cond

# (A) validation: E/(E+T) monotonically predicts the realized ratio across the 3 properties
v = r["validation"]
chk("E/(E+T) order is stability<density<gap", v["order_by_efrac"] == ["stability", "density", "gap"])
chk("E/(E+T) monotonically predicts C-14 ratio", v["monotone_efrac_predicts_ratio"] is True)
chk("Spearman(n=3) == 1.0 (perfect rank agreement)", abs(v["spearman_n3"] - 1.0) < 1e-6)

# (B) selection: key_quality = between_frac * carryover_r2, and recipe = argmax
sel = r["selection"]; tbl = sel["table"]; rec = sel["recipe"]
for prop, scores in tbl.items():
    for kn, s in scores.items():
        chk(f"{prop}/{kn} key_quality = bf*R2",
            abs(s["key_quality"] - s["between_frac"] * s["carryover_r2"]) < 1e-3)
    best = max(scores, key=lambda k: scores[k]["key_quality"])
    chk(f"{prop} recipe picks argmax key ({best})", rec[prop]["best_key"] == best)

# the recipe validates the operative key for density & stability
chk("recipe recommends els_nat for density", rec["density"]["best_key"] == "els_nat")
chk("recipe recommends els_nat for e_above_hull", rec["e_above_hull"]["best_key"] == "els_nat")
# and the coarse anion key collapses density's between-fraction (the honest caveat)
chk("anion key collapses density between_frac (<0.4)", tbl["density"]["anion"]["between_frac"] < 0.4)

print("\nCE18-VERIFY:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
