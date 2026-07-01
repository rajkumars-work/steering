#!/usr/bin/env python3
"""C-E3: baselines for the joint (wide-gap AND stable) — telling vs showing, with CIs.

Three conditions, matched sample size, >=3 seeds:
  1. telling_naive   — chemistry drawn from the FULL distribution, prompt tagged for both
                       (bg-vhigh hull-vlow). "ask for both at once."
  2. telling_strong  — chemistry drawn from the model's wide-gap-associated prior (chemistries
                       that appear with bg-{high,vhigh} tags in training), same both-tags prompt.
                       The strongest single-request setting.
  3. showing         — chemistry drawn from chemistries whose DATA structures are BOTH wide-gap
                       and stable (selection), naked prompt. "show, don't tell."

Joint hit = (surrogate band gap >= GAP_THR) AND (MACE e_above_hull <= EAH_STABLE).
Gap: composition XGBoost surrogate (the same DFT-band-gap family the bg-tags were labelled from;
the structural MLIP gap needs the heavy Ray/Dielectrics stack and is impractical to run here).
Stability: single-MACE e_above_hull. Report hit-rate +/- Wilson & bootstrap CI per condition.

Output: review1/ce3_joint.json (incremental). Detached, multi-hour.
"""
import sys, os, json, time, csv, random, importlib.util, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import CKPT, VERSION, DEVICE, require_train_csv, BG_MODEL, EPS0_PATH, OUTDIR  # repo-relative; sets sys.path
import numpy as np
import xgboost as xgb

TRAIN_CSV=require_train_csv()
HERE=OUTDIR; OUT=os.path.join(HERE,"ce3_joint.json")
SEEDS=[0,1,2]; N_C=120; GAP_THR=3.0; EAH_STABLE=0.1; EAH_MAX=5.0
TELL_TAGS="bg-vhigh hull-vlow"     # ask for wide gap + stable

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)

# composition-feature fn imported directly (package __init__ pulls dscribe, which is absent)
_spec=importlib.util.spec_from_file_location("eps0_direct",EPS0_PATH)
_eps0=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_eps0)
compute_composition_features=_eps0.compute_composition_features

def wilson(k,n,z=1.96):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(c-h,4),round(c+h,4))

def main():
    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc

    # ---- build chemistry pools from training rows ----
    full=defaultdict(int); widegap=defaultdict(int); joint=defaultdict(int)
    with open(TRAIN_CSV) as f:
        rd=csv.reader(f); next(rd)
        for row in rd:
            if len(row)<6: continue
            parts=row[0].split("|")
            if len(parts)<3: continue
            els=parts[0].strip(); nat=parts[1].strip(); tags=parts[2]
            key=(els,nat); full[key]+=1
            wide = ("bg-high" in tags) or ("bg-vhigh" in tags)
            # the dataset is all low-hull, so the e_above_hull column is ~uniformly small and
            # does NOT discriminate; the hull-vlow *tag* (most-stable bin within the set) does.
            stable = ("hull-vlow" in tags)
            if wide: widegap[key]+=1
            if wide and stable: joint[key]+=1
    def pool(d):
        ks=list(d); w=[d[k] for k in ks]; return ks,w
    pools={"telling_naive":pool(full),"telling_strong":pool(widegap),"showing":pool(joint)}
    log(f"pools: full {len(full)}  widegap {len(widegap)}  joint(widegap&stable) {len(joint)}")

    model,sp=load_model(CKPT,DEVICE); calc=load_stability_calc(device=DEVICE)
    bg=xgb.Booster(); bg.load_model(BG_MODEL)

    def gap_of(atoms):
        try:
            feat=compute_composition_features(atoms); dm=xgb.DMatrix(feat.reshape(1,-1))
            return max(0.0,float(bg.predict(dm)[0]))
        except Exception: return None
    def eah_of(atoms):
        try:
            e=compute_e_above_hull(atoms,calc=calc,timeout=120).get("e_above_hull")
            return e if (e is not None and np.isfinite(e) and abs(e)<=EAH_MAX) else None
        except Exception: return None

    res={"config":{"N_per_condition":N_C,"seeds":SEEDS,"gap_thr":GAP_THR,"eah_stable":EAH_STABLE,
                   "gap":"xgb composition DFT band-gap surrogate","stability":"single-MACE e_above_hull",
                   "tell_tags":TELL_TAGS},"conditions":{}}
    for cond,(keys,w) in pools.items():
        tags = "" if cond=="showing" else TELL_TAGS
        per_seed=[]; recs=[]
        for seed in SEEDS:
            rng=random.Random(7000+seed)
            picks=rng.choices(keys,weights=w,k=N_C)
            hits=0; n=0; ng=0; ns=0
            for (els,nat) in picks:
                prompt=f"{els} | {nat} | {tags}".rstrip()+" "
                try:
                    o=generate_one(model,sp,prompt,"alex",VERSION,DEVICE,top_k=10)
                    a=getattr(o,"atoms",None)
                    if a is None: continue
                except Exception: continue
                g=gap_of(a); e=eah_of(a)
                if g is None or e is None: continue
                n+=1; wide=g>=GAP_THR; stab=e<=EAH_STABLE
                ng+=int(wide); ns+=int(stab); hits+=int(wide and stab)
                recs.append({"seed":seed,"els":els,"nat":nat,"gap":round(g,3),"eah":round(e,4),
                             "wide":wide,"stable":stab,"joint":bool(wide and stab)})
            per_seed.append({"seed":seed,"n":n,"hits":hits,
                             "joint_rate":round(hits/n,4) if n else None,
                             "wide_rate":round(ng/n,4) if n else None,
                             "stable_rate":round(ns/n,4) if n else None})
            log(f"  {cond} seed {seed}: joint {hits}/{n} = {round(hits/max(n,1),3)} (wide {ng}, stable {ns})")
            json.dump(res|{"_partial_recs":len(recs)},open(OUT,"w"),indent=2)
        tot_h=sum(s["hits"] for s in per_seed); tot_n=sum(s["n"] for s in per_seed)
        rates=[s["joint_rate"] for s in per_seed if s["joint_rate"] is not None]
        res["conditions"][cond]={"per_seed":per_seed,"pooled_hits":tot_h,"pooled_n":tot_n,
            "joint_rate":round(tot_h/tot_n,4) if tot_n else None,
            "wilson_ci95":wilson(tot_h,tot_n),
            "seed_mean":round(float(np.mean(rates)),4) if rates else None,
            "seed_std":round(float(np.std(rates)),4) if rates else None,
            "tags":tags,"n_chemistries_in_pool":len(keys),"records":recs}
        json.dump(res,open(OUT,"w"),indent=2)
    log("=== joint hit-rates (wide-gap & stable) ===")
    for cond in pools:
        c=res["conditions"][cond]; log(f"  {cond:16s} {c['joint_rate']}  Wilson95 {c['wilson_ci95']}  (n={c['pooled_n']})")
    print("CE3-JOINT-DONE")

if __name__=="__main__": main()
