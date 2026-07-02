#!/usr/bin/env python3
"""Verify C-E17's own claim: under the CI-lower carry-over, stability's data-lever stays a large
multiple of the tag-knob, and the discount math is self-consistent. Pure CPU; no GPU/checkpoint.
Run: python verify_ce17.py   (after ce17_stability_bound.py has written ce17_stability_bound.json)."""
import os, json, math, sys

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "ce17_stability_bound.json")))
inp = r["inputs"]; ok = True

def chk(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), name); ok = ok and cond

# 1. observed ratio matches bin/knob
chk("observed ratio = bin/knob", abs(inp["ratio_observed"] - inp["bin_shift"]/inp["knob_shift"]) < 0.1)

# 2. primary discount factor = sqrt(R2_lower/R2_point) for single-MACE
c = inp["carryover"]["single_mace_ce2"]
exp_fac = math.sqrt(c["ci95"][0] / c["r2"])
got_fac = r["bounds"]["single_mace_ce2"]["models"]["attenuation_sqrtR2"]["discount_factor"]
chk("sqrtR2 discount factor correct", abs(exp_fac - got_fac) < 1e-3)

# 3. principled floor is the sqrtR2 model vs point knob, and is >= the variance model (less strict)
m = r["bounds"]["single_mace_ce2"]["models"]
chk("sqrtR2 floor >= variance floor (attenuation is less strict than variance)",
    m["attenuation_sqrtR2"]["ratio_vs_point_knob"] >= m["variance_R2"]["ratio_vs_point_knob"])
chk("variance floor >= raw floor", m["variance_R2"]["ratio_vs_point_knob"] >= m["raw_R2lower"]["ratio_vs_point_knob"])

# 4. the CLAIM: even at CI-lower carry-over, floor is a large multiple (>= 7x under every model vs point knob)
floors = [m[k]["ratio_vs_point_knob"] for k in ("attenuation_sqrtR2", "variance_R2", "raw_R2lower")]
chk("all models >= 7x vs point knob (data-lever survives worst-case carry-over)", min(floors) >= 7.0)
chk("headline floor is the principled sqrtR2 value (~24x)", abs(r["headline"]["floor_ratio"] - 24.3) < 0.5)

# 5. honesty guard: sub-1x only arises with the worst-case knob upper-CI, whose CI includes 0
chk("tag-knob CI includes 0 (so worst-case-knob ratio is not a genuine telling win)",
    inp["knob_ci95"][0] <= 0 <= inp["knob_ci95"][1])

print("\nCE17-VERIFY:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
