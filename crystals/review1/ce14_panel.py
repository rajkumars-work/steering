#!/usr/bin/env python3
"""C-E14 REVISED (per 2026-06-29 revision): bins-vs-knobs over a PROPERTY PANEL, plotting the
bin/knob shift ratio against the between-fraction E/(E+T). Budget says showing's reach ∝ √(χ²·E),
telling's ∝ √(χ²·T), so the ratio should rise with E/(E+T).

Panel = the properties with a VARYING property-tag (a usable 'knob'): band gap (bg-*), density
(rho-*), stability/e_above_hull (hull-*). (Dielectric k-* is uniformly k-mid -> no knob; excluded.)
Per property: knob shift = |mean_P(broad + tag-hi) - mean_P(broad + tag-lo)| (chemistry fixed, tag
flipped); bin shift = |mean_P(high-P chems) - mean_P(low-P chems)| (chemistry selection, no tag).
ratio = bin/knob. E/(E+T) from the C-E2 budget audit (density, stability) or computed here (gap).
N=80 x 3 seeds, bootstrap CIs. gap=surrogate, density=geometry (CPU); stability=single-MACE.
Output: review1/ce14_panel.json. Detached.
"""
import sys, os, json, time, csv, random, importlib.util
from collections import defaultdict
sys.path[:0] = ["/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed",
                "/home/ubuntu/packages/lemat-genbench/src"]
import numpy as np, xgboost as xgb

CKPT = "/opt/dlami/nvme/recast/train/mix_ep120_ckpt"; VERSION = "d15_binrho_k7"; DEVICE = "cuda"
TRAIN = os.environ.get("CUES_TRAIN_CSV", "/opt/dlami/nvme/recast/train/alex_nolemat_lowhull_dataset_train.csv")
PD = "/home/ubuntu/code/py/dielectric/pipeline/data"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ce14_panel.json")
N_C, SEEDS, MIN_MEMBERS, N_AUDIT, M_AUDIT, TOPK = 80, [0, 1, 2], 5, 500, 6, 40
AMU = 1.6605390666
EAH_MAX = 5.0
# property panel: (name, tag_lo, tag_hi, needs_mace)
PANEL = [("gap", "bg-vlow", "bg-vhigh", False),
         ("density", "rho-vlow", "rho-vhigh", False),
         ("stability", "hull-high", "hull-vlow", True)]   # 'high stability' = low e_above_hull = hull-vlow

_sp = importlib.util.spec_from_file_location("e", "/home/ubuntu/code/py/dielectric/chem/surrogates/eps0.py")
_e = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_e); ccf = _e.compute_composition_features
from dielectric_data.reader import parse_target
_bg = xgb.Booster(); _bg.load_model(f"{PD}/xgb_composition_dft_band_gap.json")
_calc = {"c": None}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def prop_val(name, a):
    if name == "gap":
        try: return max(0.0, float(_bg.predict(xgb.DMatrix(ccf(a).reshape(1, -1)))[0]))
        except Exception: return None
    if name == "density":
        try:
            v = float(a.get_volume()); return float(np.sum(a.get_masses())) / v * AMU if v > 0 else None
        except Exception: return None
    if name == "stability":
        try:
            from chem.stability import compute_e_above_hull
            e = compute_e_above_hull(a, calc=_calc["c"], timeout=120).get("e_above_hull")
            return e if (e is not None and np.isfinite(e) and abs(e) <= EAH_MAX) else None
        except Exception: return None

def ETp(name, peritem):
    """E/(E+T) from the C-E2 per-item data (within T = mean per-chem var; between E = var per-chem means)."""
    means, vars = [], []
    for s in peritem:
        for b in peritem[s].values():
            key = name if name != "stability" else "e_above_hull"
            vals = [r[key] for r in b["data"] if r.get(key) is not None]
            if len(vals) >= 2: means.append(np.mean(vals)); vars.append(np.var(vals))
    if len(means) < 3: return None
    T, E = float(np.mean(vars)), float(np.var(means)); return round(E / (E + T), 3) if (E + T) else None


def main():
    from eval.screening import load_model, generate_one
    from chem.stability import load_stability_calc
    bins = defaultdict(list)
    for row in csv.reader(open(TRAIN)):
        if not row or row[0] == "source": continue
        s = row[0].split("|")
        if len(s) >= 2: bins[(s[0].strip(), s[1].strip())].append(row)
    elig = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
    audit = random.Random(7).sample(elig, min(N_AUDIT, len(elig)))
    peritem = json.load(open(os.path.join(HERE, "ce2_peritem.json")))["peritem"]

    # per-chem data means for bin selection: gap & density from audit (CPU); stability from C-E2 peritem
    log(f"auditing {len(audit)} chems for gap/density bin ranks ...")
    gmean, dmean = {}, {}
    for (els, nat) in audit:
        gs, ds = [], []
        for r in random.Random(7).sample(bins[(els, nat)], min(M_AUDIT, len(bins[(els, nat)]))):
            try: a = parse_target(r[1], VERSION, r[3])
            except Exception: a = None
            if a is None: continue
            g = prop_val("gap", a); d = prop_val("density", a)
            if g is not None: gs.append(g)
            if d is not None: ds.append(d)
        if len(gs) >= 3: gmean[(els, nat)] = float(np.mean(gs))
        if len(ds) >= 3: dmean[(els, nat)] = float(np.mean(ds))
    smean = {}
    for s in peritem:
        for b in peritem[s].values():
            vals = [r["e_above_hull"] for r in b["data"] if r.get("e_above_hull") is not None]
            if len(vals) >= 2: smean.setdefault((b["elements"], b["natoms"]), []).extend(vals)
    smean = {k: float(np.mean(v)) for k, v in smean.items() if len(v) >= 3}
    rankmaps = {"gap": gmean, "density": dmean, "stability": smean}

    model, sp = load_model(CKPT, DEVICE)

    def gen_score(name, srcs):
        out = []
        for s in srcs:
            try:
                o = generate_one(model, sp, s, "alex", VERSION, DEVICE, top_k=10)
                a = getattr(o, "atoms", None)
                if a is None: continue
                v = prop_val(name, a)
                if v is not None: out.append(v)
            except Exception:
                pass
        return out

    res = {"N": N_C, "seeds": SEEDS, "topk": TOPK, "panel": {}}
    for (name, tlo, thi, needs_mace) in PANEL:
        if needs_mace and _calc["c"] is None:
            log("loading single-MACE for stability ..."); _calc["c"] = load_stability_calc(device=DEVICE)
        ranked = sorted(rankmaps[name], key=rankmaps[name].get)
        low_ch, high_ch = ranked[:TOPK], ranked[-TOPK:]
        knob_hi, knob_lo, bin_hi, bin_lo = [], [], [], []
        for seed in SEEDS:
            rng = random.Random(1400 + seed)
            broad = rng.choices(audit, weights=[len(bins[k]) for k in audit], k=N_C)
            knob_hi += gen_score(name, [f"{e} | {n} | {thi} " for (e, n) in broad])
            knob_lo += gen_score(name, [f"{e} | {n} | {tlo} " for (e, n) in broad])
            bin_hi += gen_score(name, [f"{e} | {n} | " for (e, n) in rng.choices(high_ch, weights=[len(bins[k]) for k in high_ch], k=N_C)])
            bin_lo += gen_score(name, [f"{e} | {n} | " for (e, n) in rng.choices(low_ch, weights=[len(bins[k]) for k in low_ch], k=N_C)])
            log(f"  {name} seed{seed}: knob hi/lo {np.mean(knob_hi):.3f}/{np.mean(knob_lo):.3f} bin hi/lo {np.mean(bin_hi):.3f}/{np.mean(bin_lo):.3f}")
        def shift_boot(hi, lo):
            H, L = np.array(hi), np.array(lo); rngb = np.random.default_rng(0); b = []
            for _ in range(5000): b.append(abs(H[rngb.integers(0,len(H),len(H))].mean() - L[rngb.integers(0,len(L),len(L))].mean()))
            return abs(float(H.mean()-L.mean())), b
        ks, kb = shift_boot(knob_hi, knob_lo); bs, bb = shift_boot(bin_hi, bin_lo)
        rngb = np.random.default_rng(2); ratios = [bb[i]/kb[i] for i in rngb.integers(0, 5000, 5000) if kb[i] > 1e-9]
        res["panel"][name] = {"E_over_EplusT": ETp(name, peritem), "knob_shift": round(ks, 4), "bin_shift": round(bs, 4),
                              "bin_over_knob": round(bs/ks, 2) if ks > 1e-9 else None,
                              "ratio_ci95": [round(float(np.percentile(ratios,2.5)),2), round(float(np.percentile(ratios,97.5)),2)] if ratios else None,
                              "knob_ci95": [round(float(np.percentile(kb,2.5)),4), round(float(np.percentile(kb,97.5)),4)],
                              "bin_ci95": [round(float(np.percentile(bb,2.5)),4), round(float(np.percentile(bb,97.5)),4)]}
        json.dump(res, open(OUT, "w"), indent=2)
        log(f"  == {name}: E/(E+T)={res['panel'][name]['E_over_EplusT']} bin/knob={res['panel'][name]['bin_over_knob']} {res['panel'][name]['ratio_ci95']}")
    log("=== C-E14 panel: bin/knob ratio vs E/(E+T) ===")
    for name in [p[0] for p in PANEL]:
        r = res["panel"][name]; log(f"  {name:10s} E/(E+T)={r['E_over_EplusT']}  bin/knob={r['bin_over_knob']} CI{r['ratio_ci95']}  (knob {r['knob_shift']}, bin {r['bin_shift']})")
    print("CE14-PANEL-DONE")


if __name__ == "__main__":
    main()
