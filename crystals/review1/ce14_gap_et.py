#!/usr/bin/env python3
"""Post-hoc: compute the band-gap E/(E+T) (between-chemistry / total variance) for the C-E14 panel.
ce2_peritem.json carries density/bvs_gii/e_above_hull per item but NOT gap, so gap's E/(E+T) came out
None. Here we decode the SAME audit chemistries (seed 7, MIN_MEMBERS>=5) from the training CSV, score
each row's gap with the XGBoost composition surrogate, and form T = mean within-chem var, E = var of
per-chem means -> E/(E+T). Mirrors panel.ETp(). Writes ce14_gap_et.json."""
import sys, os, json, csv, random, importlib.util
from collections import defaultdict
sys.path[:0] = ["/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed"]
import numpy as np, xgboost as xgb

VERSION = "d15_binrho_k7"
TRAIN = os.environ.get("CUES_TRAIN_CSV", "/opt/dlami/nvme/recast/train/alex_nolemat_lowhull_dataset_train.csv")
PD = "/home/ubuntu/code/py/dielectric/pipeline/data"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ce14_gap_et.json")
MIN_MEMBERS, N_AUDIT, M_AUDIT = 5, 500, 6   # same as ce14_panel

_sp = importlib.util.spec_from_file_location("e", "/home/ubuntu/code/py/dielectric/chem/surrogates/eps0.py")
_e = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_e); ccf = _e.compute_composition_features
from dielectric_data.reader import parse_target
_bg = xgb.Booster(); _bg.load_model(f"{PD}/xgb_composition_dft_band_gap.json")

def gap(a):
    try: return max(0.0, float(_bg.predict(xgb.DMatrix(ccf(a).reshape(1, -1)))[0]))
    except Exception: return None

bins = defaultdict(list)
for row in csv.reader(open(TRAIN)):
    if not row or row[0] == "source": continue
    s = row[0].split("|")
    if len(s) >= 2: bins[(s[0].strip(), s[1].strip())].append(row)
elig = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
audit = random.Random(7).sample(elig, min(N_AUDIT, len(elig)))

means, varis = [], []
for (els, nat) in audit:
    gs = []
    for r in random.Random(7).sample(bins[(els, nat)], min(M_AUDIT, len(bins[(els, nat)]))):
        try: a = parse_target(r[1], VERSION, r[3])
        except Exception: a = None
        if a is None: continue
        g = gap(a)
        if g is not None: gs.append(g)
    if len(gs) >= 2: means.append(float(np.mean(gs))); varis.append(float(np.var(gs)))

T, E = float(np.mean(varis)), float(np.var(means))
et = round(E / (E + T), 3) if (E + T) else None
out = {"property": "gap", "n_chems": len(means), "within_T": round(T, 4), "between_E": round(E, 4),
       "E_over_EplusT": et}
json.dump(out, open(OUT, "w"), indent=2)
print(json.dumps(out))
print("CE14-GAP-ET-DONE")
