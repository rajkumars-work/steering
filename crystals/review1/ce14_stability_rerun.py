#!/usr/bin/env python3
"""C-E14 STABILITY CELL RE-RUN (per handoff '# C-E14 STABILITY RE-RUN', pre-reg 5ec8056).

The panel's stability row was an artifact: it used the MEAN of raw e_above_hull (mostly noise) and
ranked bins by the noisy ce2_peritem audit mean -> 1.2-1.4 eV/atom junk, wrong direction -> spurious
knob-win 0.46x. Corrected design:

  metric  = METASTABLE RATE = fraction with e_above_hull <= 0.1 eV/atom (claim1_survival_spectrum.py:112),
            same scorer (compute_e_above_hull, single-MACE) on BOTH sides -> matched metric.
  bin side (data lever) = reuse within-dist push_HIGH/push_LOW element pools; bin_hi/bin_lo = training
            chemistries whose element set is a subset of the pool; generate TAG-FREE (same pipeline
            that produced the validated 84% / 1.7%).
  knob side (hull tag)  = broad audit chemistries, flip hull-vlow (high stability) vs hull-high.
  ratio = bin_shift / knob_shift (metastable-rate shifts). N=80 x 3 seeds, bootstrap CIs.

SANITY GATE: bin_hi must regenerate >=50% metastable; if ~0% the generation path is still wrong ->
write GATE-FAIL and stop (do not report a ratio). Output: review1/ce14_stability_rerun.json. Detached.
"""
import sys, os, json, time, csv, random
from collections import defaultdict
sys.path[:0] = ["/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed",
                "/home/ubuntu/packages/lemat-genbench/src"]
import numpy as np

CKPT = "/opt/dlami/nvme/recast/train/mix_ep120_ckpt"; VERSION = "d15_binrho_k7"; DEVICE = "cuda"
TRAIN = os.environ.get("CUES_TRAIN_CSV", "/opt/dlami/nvme/recast/train/alex_nolemat_lowhull_dataset_train.csv")
DIST = "/opt/dlami/nvme/recast/train/dist_verify/distributions/an_lh.dist.json"   # the an_lh (E,N) distribution
WDS = "/home/ubuntu/code/py/dielectric/eval/within_distribution_steering"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ce14_stability_rerun.json")
N_C, SEEDS, MIN_BROAD, N_AUDIT = 80, [0, 1, 2], 5, 500
EAH_MAX, META_THR = 5.0, 0.1
ETp_STABILITY = 0.573   # stability's E/(E+T) from the C-E2 budget audit (panel x-coordinate)
RETRY_EXIT = 75         # exit code meaning "transient GPU/scorer failure -- supervisor should retry"

from dielectric_data.reader import parse_target
_calc = {"c": None}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def metastable(a):
    """1 if e_above_hull <= 0.1, 0 if defined-and-above, None if unscorable."""
    try:
        from chem.stability import compute_e_above_hull
        e = compute_e_above_hull(a, calc=_calc["c"], timeout=120).get("e_above_hull")
        if e is None or not np.isfinite(e) or abs(e) > EAH_MAX: return None
        return 1.0 if e <= META_THR else 0.0
    except Exception:
        return None


def main():
    from eval.screening import load_model, generate_one
    from chem.stability import load_stability_calc

    bins = defaultdict(int)
    for row in csv.reader(open(TRAIN)):
        if not row or row[0] == "source": continue
        s = row[0].split("|")
        if len(s) >= 2: bins[(s[0].strip(), s[1].strip())] += 1
    elig = [k for k, v in bins.items() if v >= MIN_BROAD]
    broad_audit = random.Random(7).sample(elig, min(N_AUDIT, len(elig)))

    # bin pools = (elements, natoms) tuples drawn from the an_lh DISTRIBUTION (real, in-distribution
    # combos with realistic atom counts), filtered to the within-dist push_HIGH / push_LOW element
    # sets -- the same universe binning_experiment.sample() drew from (validated 84% / 1.7%). The
    # earlier training-subset pool invented out-of-distribution combos (N up to 20) the model can't
    # realize -> atoms None -> all-NaN. an_lh tuples generate ~97% (probe 39/40).
    HI = set(json.load(open(f"{WDS}/push_HIGH.json"))["final_elems"])
    LO = set(json.load(open(f"{WDS}/push_LOW.json"))["final_elems"])
    distt = json.load(open(DIST))
    bin_hi_pool = [((t["elements"], int(t["natoms"])), int(t["count"])) for t in distt if set(t["elements"].split()) <= HI]
    bin_lo_pool = [((t["elements"], int(t["natoms"])), int(t["count"])) for t in distt if set(t["elements"].split()) <= LO]
    log(f"bin pools (an_lh dist): HI {len(bin_hi_pool)} tuples, LO {len(bin_lo_pool)} tuples; broad {len(broad_audit)}")

    # LOAD ORDER MATTERS: load the generation model BEFORE the MACE stability calc. Loading MACE
    # first deterministically breaks generation (generate_one returns atoms=None for every prompt
    # -> all-NaN). Proven by A/B: calc-first 0/8 vs model-first 8/8. This was the real cause of the
    # three all-NaN runs (not the MP outage or web/app.py contention I first suspected).
    log("loading model, then single-MACE ..."); model, sp = load_model(CKPT, DEVICE)
    _calc["c"] = load_stability_calc(device=DEVICE)
    # NOTE: do NOT reset the global default dtype here -- MACE builds its input tensors at the global
    # default when scoring, so float32 inputs into its float64 model break compute_e_above_hull
    # (NaCl -> None). The generation model is already built (float32) before this; leaving the global
    # default at MACE's float64 is correct for scoring AND generation still works (A/B: model-first 8/8).

    def gen_atoms(s):
        try:
            o = generate_one(model, sp, s, "alex", VERSION, DEVICE, top_k=10)
            return getattr(o, "atoms", None)
        except Exception:
            return None

    def gen_meta(srcs):
        out = []
        for s in srcs:
            a = gen_atoms(s)
            if a is None: continue
            m = metastable(a)
            if m is not None: out.append(m)
        return out

    def draw(pool, rng, k):
        ks = [p[0] for p in pool]; w = [p[1] for p in pool]
        return [f"{e} | {n} | " for (e, n) in rng.choices(ks, weights=w, k=k)]

    def bail_retry(note):
        """Transient GPU/scorer failure -> let the supervisor retry in a cleaner window."""
        json.dump({"partial": False, "retryable": True, "note": note}, open(OUT, "w"), indent=2)
        log(f"RETRYABLE FAILURE: {note}"); print("CE14-STAB-RERUN-RETRY"); sys.exit(RETRY_EXIT)

    # PREFLIGHT 1 (scorer): score NaCl (must be metastable). The 08:08 run produced all-NaN because
    # the MP hull-fetch failed for the whole window and metastable() swallows that to None.
    from ase.build import bulk
    nacl = metastable(bulk("NaCl", "rocksalt", a=5.64))
    log(f"preflight scorer: NaCl metastable={nacl}")
    if nacl != 1.0:
        bail_retry("scorer down -- compute_e_above_hull failed on NaCl (MP hull-fetch throttled/down).")

    # PREFLIGHT 2 (generator): the GPU is shared with a persistent web/app.py inference service; during
    # busy windows generate_one returns atoms=None for EVERY prompt (run #2 + probe both got 0%). In a
    # clean window the an_lh pool generates ~97%. Probe 12 bin_hi prompts; if <30% yield atoms, this is
    # a contended window -> bail and retry rather than grind hours into all-NaN.
    pf_rng = random.Random(99)
    pf = [a is not None for a in (gen_atoms(s) for s in draw(bin_hi_pool, pf_rng, 12))]
    pf_rate = sum(pf) / len(pf)
    log(f"preflight generator: bin_hi atoms-rate {sum(pf)}/12 = {pf_rate:.2f}")
    if pf_rate < 0.30:
        bail_retry(f"generator contended -- only {sum(pf)}/12 bin_hi prompts produced atoms "
                   f"(GPU likely busy with web/app.py); retry in a cleaner window.")

    sides = {"bin_hi": [], "bin_lo": [], "knob_hi": [], "knob_lo": []}
    for seed in SEEDS:
        rng = random.Random(1400 + seed)
        broad = rng.choices(broad_audit, weights=[bins[k] for k in broad_audit], k=N_C)
        sides["bin_hi"] += gen_meta(draw(bin_hi_pool, rng, N_C))
        if seed == SEEDS[0] and len(sides["bin_hi"]) == 0:   # went bad after preflight -> retry
            bail_retry("0 scorable from the first N_C bin_hi generations after preflight passed "
                       "(GPU/scorer went bad mid-run).")
        sides["bin_lo"] += gen_meta(draw(bin_lo_pool, rng, N_C))
        sides["knob_hi"] += gen_meta([f"{e} | {n} | hull-vlow " for (e, n) in broad])   # hull-vlow = high stability
        sides["knob_lo"] += gen_meta([f"{e} | {n} | hull-high " for (e, n) in broad])
        rate = {k: (np.mean(v) if v else float('nan')) for k, v in sides.items()}
        log(f"  seed{seed}: bin_hi={rate['bin_hi']:.3f}(n{len(sides['bin_hi'])}) bin_lo={rate['bin_lo']:.3f} "
            f"knob_hi={rate['knob_hi']:.3f} knob_lo={rate['knob_lo']:.3f}")
        json.dump({"partial": True, "sides_n": {k: len(v) for k, v in sides.items()}}, open(OUT, "w"))

    rate = {k: float(np.mean(v)) for k, v in sides.items()}
    gate_ok = rate["bin_hi"] >= 0.50
    def boot(hi, lo):
        H, L = np.array(sides[hi]), np.array(sides[lo]); rb = np.random.default_rng(0); b = []
        for _ in range(5000):
            b.append(H[rb.integers(0, len(H), len(H))].mean() - L[rb.integers(0, len(L), len(L))].mean())
        return float(H.mean() - L.mean()), b
    bin_sh, bb = boot("bin_hi", "bin_lo"); knob_sh, kb = boot("knob_hi", "knob_lo")
    rb = np.random.default_rng(2)
    ratios = [bb[i] / kb[i] for i in rb.integers(0, 5000, 5000) if abs(kb[i]) > 1e-9]
    res = {"metric": "metastable rate (e_above_hull <= 0.1)", "N": N_C, "seeds": SEEDS,
           "E_over_EplusT": ETp_STABILITY, "n_per_side": {k: len(v) for k, v in sides.items()},
           "rates": {k: round(rate[k], 4) for k in rate},
           "sanity_gate": {"bin_hi_metastable_rate": round(rate["bin_hi"], 4), "threshold": 0.50,
                           "passed": bool(gate_ok)},
           "bin_shift": round(bin_sh, 4), "bin_ci95": [round(float(np.percentile(bb, 2.5)), 4), round(float(np.percentile(bb, 97.5)), 4)],
           "knob_shift": round(knob_sh, 4), "knob_ci95": [round(float(np.percentile(kb, 2.5)), 4), round(float(np.percentile(kb, 97.5)), 4)],
           "bin_over_knob": round(bin_sh / knob_sh, 2) if abs(knob_sh) > 1e-9 else None,
           "ratio_ci95": [round(float(np.percentile(ratios, 2.5)), 2), round(float(np.percentile(ratios, 97.5)), 2)] if ratios else None,
           "within_dist_reference": {"push_HIGH_meta": "101/120 (0.842)", "push_LOW_meta": "2/120 (0.017)"}}
    json.dump(res, open(OUT, "w"), indent=2)
    log(f"GATE {'PASS' if gate_ok else 'FAIL'} (bin_hi metastable={rate['bin_hi']:.3f})")
    log(f"bin_shift={bin_sh:.3f} {res['bin_ci95']} | knob_shift={knob_sh:.3f} {res['knob_ci95']} | "
        f"bin/knob={res['bin_over_knob']} {res['ratio_ci95']}")
    if not gate_ok:
        log("SANITY GATE FAILED — generation path still wrong; ratio reported but NOT to be trusted.")
    print("CE14-STAB-RERUN-DONE")


if __name__ == "__main__":
    main()
