#!/usr/bin/env python3
"""C-E12 pair-selection (data-side, no GPU): pick two ANTI-correlated properties for the
trade-off-frontier coverage experiment, AND check which are conditionable via training tags.

Note: in the CUES training data the dielectric tag `k-*` is uniformly `k-mid` (no variation), so the
model has NO dielectric conditioning — the gap∧dielectric pair cannot have a real multi-property
*conditioning* baseline (method 2). The varying/conditionable tags are bg-* (gap), hull-* (stability),
rho-* (density). So we evaluate candidate anti-correlated pairs and prefer one where BOTH halves are
conditionable (so method-2 is genuine): band gap ∧ density.
"""
import sys, os, json, csv, random, importlib.util
sys.path[:0] = ["/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed"]
import numpy as np, xgboost as xgb

TRAIN = "/opt/dlami/nvme/recast/train/alex_nolemat_lowhull_dataset_train.csv"
PD = "/home/ubuntu/code/py/dielectric/pipeline/data"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ce12_pairselect.json")
AMU = 1.6605390666
sp = importlib.util.spec_from_file_location("e", "/home/ubuntu/code/py/dielectric/chem/surrogates/eps0.py")
e = importlib.util.module_from_spec(sp); sp.loader.exec_module(e); ccf = e.compute_composition_features
from dielectric_data.reader import parse_target
bg = xgb.Booster(); bg.load_model(f"{PD}/xgb_composition_dft_band_gap.json")
ep = xgb.Booster(); ep.load_model(f"{PD}/xgb_composition_dft_eps_0.json")


def main():
    bins = {}
    for row in csv.reader(open(TRAIN)):
        if not row or row[0] == "source": continue
        s = row[0].split("|")
        if len(s) >= 2: bins.setdefault((s[0].strip(), s[1].strip()), []).append(row)
    elig = [k for k, v in bins.items() if len(v) >= 12]
    chosen = random.Random(7).sample(elig, min(50, len(elig)))
    G, E, D = [], [], []
    for (els, nat) in chosen:
        gs, es, ds = [], [], []
        for r in random.Random(7).sample(bins[(els, nat)], 10):
            try: a = parse_target(r[1], "d15_binrho_k7", r[3])
            except Exception: a = None
            if a is None: continue
            f = ccf(a).reshape(1, -1)
            gs.append(max(0.0, float(bg.predict(xgb.DMatrix(f))[0])))
            es.append(max(1.0, float(ep.predict(xgb.DMatrix(f))[0])))
            try:
                v = float(a.get_volume()); ds.append(float(np.sum(a.get_masses())) / v * AMU if v > 0 else np.nan)
            except Exception: ds.append(np.nan)
        if gs and es and ds:
            G.append(np.mean(gs)); E.append(np.mean(es)); D.append(np.nanmean(ds))
    G, E, D = np.array(G), np.array(E), np.array(D)
    def pct(x): return [round(float(np.percentile(x, p)), 2) for p in (20, 40, 60, 80)]
    res = {"n_chem": len(G),
           "corr": {"gap_eps0": round(float(np.corrcoef(G, E)[0, 1]), 3),
                    "gap_density": round(float(np.corrcoef(G, D)[0, 1]), 3),
                    "density_eps0": round(float(np.corrcoef(D, E)[0, 1]), 3)},
           "pctiles_20_40_60_80": {"gap": pct(G), "eps0": pct(E), "density": pct(D)}}
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2)); print("CE12-PAIRSELECT-DONE")


if __name__ == "__main__":
    main()
