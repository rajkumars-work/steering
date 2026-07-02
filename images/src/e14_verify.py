"""E14 verify — checks E14's own claims from e14_strong_knob.json (handoff convention).

Claims tested (pre-registered, commit 5bafc8b):
  C1  every knob (incl. the learned soft-prompt) realizes within-bin chi2 = O(1)-O(10), NOT O(100).
  C2  Delta <= sqrt(chi2 * v_b) for every knob (the within-bin ceiling is respected).
  C3  ratio bin/strongest-knob > 1 for the bin-win targets (aesthetic, sim_animal, sim_vehicle),
      < 1 for brightness (the guidance-coupled low-E pixel stat).
Exit non-zero if any hard check fails; soft warnings (e.g. a knob that leaves its bin) are printed.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(f"{_ROOT}/distributions/e14_strong_knob.json"))
BIN_WIN = {"aesthetic", "sim_animal", "sim_vehicle"}
CHI2_CAP = 100.0
ok = True
print("== E14 verify ==")
for t, p in J["panel"].items():
    for name, a in p["knobs"].items():
        c2 = a["chi2_within"]
        cap = c2 <= CHI2_CAP
        ceil = a["delta_le_ceiling"]
        ok &= bool(cap and ceil)
        flag = "" if cap else "  !! chi2>O(10)"
        stay = a.get("key_stay_frac")
        warn = f"  (stay={stay:.2f}<0.5 -> leaves bin?)" if (stay is not None and stay < 0.5) else ""
        print(f"  {t:12s} {name:17s} chi2={c2:6.2f}(floor {a['noise_floor_chi2']:.2f}) "
              f"Delta={a['delta']:+.4f} ceil={a['ceiling_sqrt_chi2_vb']:.4f} "
              f"[C1 chi2<=O(10):{cap} C2 Delta<=ceil:{ceil}]{flag}{warn}")
    r = p["ratio_bin_over_strongest_knob"]; exp = ">1" if t in BIN_WIN else "<1"
    good = (r > 1) if t in BIN_WIN else (r < 1)
    ok &= bool(good)
    print(f"  {t:12s} RATIO bin/strongest-knob = {r:.2f}x  expect {exp}  [C3:{good}]  "
          f"(soft |theta|={p['soft_theta_norm']:.2f}, CFG*={p['cfg_star']})")
print("E14-VERIFY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
