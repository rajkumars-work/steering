"""
Follow-up to claim1_survives.py: at the raw conditional (CFG=1) the model preserves the
TOTAL budget but redistributes the split (more within-bin, less between-bin). Guidance
sharpens per-class identity, so the split should move toward the data audit as CFG rises.
Sweep CFG and report where the model's within/between split matches the data's.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
CLASSES = list(range(0, 1000, 25))   # same 40 classes as claim1_survives.py
N = 32
CFGS = [2.0, 4.0]


def split(means, vars):
    return float(np.mean(vars)), float(np.var(means))


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    gd = d["g_brightness"][CLASSES]; vd = d["v_brightness"][CLASSES]
    T_data, E_data = split(gd, vd)
    print(f"DATA: within T={T_data:.5f}  between E={E_data:.5f}  total={T_data+E_data:.5f}\n")

    pipe = G.load_pipe(); L = Labeler()
    out = {"classes": CLASSES, "n_per_class": N,
           "data": {"T": T_data, "E": E_data, "V": T_data + E_data}, "sweep": {}}
    for cfg in CFGS:
        gm, gv = [], []
        for c in CLASSES:
            imgs = G.generate(pipe, [c] * N, guidance_scale=cfg, seed=2000 + c)
            b = np.asarray(L.labels(imgs)["brightness"])
            gm.append(float(b.mean())); gv.append(float(b.var()))
        gm = np.array(gm); gv = np.array(gv)
        T, En = split(gm, gv); E = En - T / N
        r = float(np.corrcoef(gd, gm)[0, 1])
        out["sweep"][f"cfg_{cfg}"] = {"T": T, "E": E, "V": T + E,
                                      "T_ratio": T / T_data, "E_ratio": E / E_data,
                                      "per_class_mean_r2": r * r}
        print(f"CFG={cfg}: within T={T:.5f} ({T/T_data:.2f}x)  between E={E:.5f} ({E/E_data:.2f}x)  "
              f"total={T+E:.5f}  r2={r*r:.3f}")
    with open(f"{OUT}/claim1_survives_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("CLAIM1-SWEEP-DONE")


if __name__ == "__main__":
    main()
