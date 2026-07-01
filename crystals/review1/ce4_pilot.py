#!/usr/bin/env python3
"""C-E4 pilot: cheap PRE-measurement signals to predict the carry-over ladder ordering
(density -> BVS-GII -> stability) BEFORE running the full C-E2 audit. Breaks the circularity:
"the budget transfers where the model is good" is only non-circular if predicted in advance.

Signals per property (computed on a SMALL, INDEPENDENT chemistry sample — seed 99, disjoint in
draw from C-E2's seed-0 set — so this is not the audit itself):
  rho      = median_chem( std_model / std_data )           within-chem spread match (1 = ideal)
  drift    = median_chem( |mean_model - mean_data| ) / global_data_std    bias
  support  = fraction of model values inside [data_min, data_max] (pooled)  support overlap

These + a loss-sensitivity argument feed a committed prediction (ce4_prediction.md).
Output: review1/ce4_pilot.json.  Small (~8 chem x 12), runs concurrently with C-E2.
"""
import sys, os, json, time, csv, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import CKPT, VERSION, DEVICE, require_train_csv, OUTDIR   # repo-relative; also sets sys.path
import numpy as np

TRAIN_CSV=require_train_csv()
HERE=OUTDIR; OUT=os.path.join(HERE,"ce4_pilot.json")
K_PILOT=8; M_DATA=12; M_GEN=12; MIN_MEMBERS=12; EAH_MAX=5.0; AMU=1.6605390666
PROPS=["density","bvs_gii","e_above_hull"]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)

def to_ase(s):
    if s is None: return None
    if hasattr(s,"get_volume"): return s
    try:
        from pymatgen.io.ase import AseAtomsAdaptor; return AseAtomsAdaptor.get_atoms(s)
    except Exception: return None

def bvs_gii(a):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
        st=a if hasattr(a,"sites") else AseAtomsAdaptor.get_structure(a)
        oxi=BVAnalyzer().get_oxi_state_decorated_structure(st); devs=[]
        for site in oxi:
            nn=oxi.get_neighbors(site,4.0)
            if nn: devs.append(calculate_bv_sum(site,nn)-site.specie.oxi_state)
        return float(np.sqrt(np.mean(np.square(devs)))) if devs else None
    except Exception: return None

def measure(s,calc,compute_eah):
    out=dict.fromkeys(PROPS,None); a=to_ase(s)
    if a is None: return out
    try:
        v=float(a.get_volume())
        if v>0: out["density"]=float(np.sum(a.get_masses()))/v*AMU
    except Exception: pass
    out["bvs_gii"]=bvs_gii(a)
    try:
        e=compute_eah(a,calc=calc,timeout=120).get("e_above_hull")
        if e is not None and np.isfinite(e) and abs(e)<=EAH_MAX: out["e_above_hull"]=float(e)
    except Exception: pass
    return out

def row_to_atoms(row):
    from dielectric_data.reader import parse_target
    try: return parse_target(row[1],VERSION,row[3])
    except Exception: return None

def main():
    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc
    bins=defaultdict(list)
    with open(TRAIN_CSV) as f:
        rd=csv.reader(f); next(rd)
        for row in rd:
            if len(row)<6: continue
            src=row[0].split("|")
            if len(src)>=2: bins[(src[0].strip(),src[1].strip())].append(row)
    eligible=[k for k,v in bins.items() if len(v)>=MIN_MEMBERS]
    chosen=random.Random(99).sample(eligible,min(K_PILOT,len(eligible)))   # independent of C-E2 (seed 0)
    log(f"pilot: {len(chosen)} independent chemistries")
    model,sp=load_model(CKPT,DEVICE); calc=load_stability_calc(device=DEVICE)
    rng=random.Random(99); D={p:[] for p in PROPS}; M={p:[] for p in PROPS}
    chem_d={p:[] for p in PROPS}; chem_m={p:[] for p in PROPS}
    for ci_,(els,nat) in enumerate(chosen):
        prompt=f"{els} | {nat} | "
        rows=rng.sample(bins[(els,nat)],min(M_DATA,len(bins[(els,nat)])))
        dvals={p:[] for p in PROPS}; mvals={p:[] for p in PROPS}
        for r in rows:
            for p,x in measure(row_to_atoms(r),calc,compute_e_above_hull).items():
                if x is not None: dvals[p].append(x)
        for _ in range(M_GEN):
            try:
                o=generate_one(model,sp,prompt,"alex",VERSION,DEVICE,top_k=10); a=getattr(o,"atoms",None)
                if a is not None:
                    for p,x in measure(a,calc,compute_e_above_hull).items():
                        if x is not None: mvals[p].append(x)
            except Exception: pass
        for p in PROPS:
            D[p]+=dvals[p]; M[p]+=mvals[p]
            if len(dvals[p])>=2 and len(mvals[p])>=2:
                chem_d[p].append((np.mean(dvals[p]),np.std(dvals[p])))
                chem_m[p].append((np.mean(mvals[p]),np.std(mvals[p])))
        log(f"  [{ci_+1}/{len(chosen)}] {els}|{nat} done")
    sig={}
    for p in PROPS:
        d=np.array(D[p]); m=np.array(M[p])
        dmu=np.array([x[0] for x in chem_d[p]]); dsd=np.array([x[1] for x in chem_d[p]])
        mmu=np.array([x[0] for x in chem_m[p]]); msd=np.array([x[1] for x in chem_m[p]])
        gstd=float(np.std(d)) if len(d) else float("nan")
        rho=float(np.median(msd/np.maximum(dsd,1e-9))) if len(dsd) else float("nan")
        drift=float(np.median(np.abs(mmu-dmu))/max(gstd,1e-9)) if len(dmu) else float("nan")
        support=float(np.mean((m>=d.min())&(m<=d.max()))) if len(d) and len(m) else float("nan")
        sig[p]={"rho_spread_ratio":round(rho,3),"drift_norm":round(drift,3),
                "support_overlap":round(support,3),"n_data":int(len(d)),"n_model":int(len(m))}
    json.dump({"chosen":chosen,"signals":sig},open(OUT,"w"),indent=2)
    log("=== pilot signals (pre-measurement) ===")
    for p in PROPS: log(f"  {p:14s} {sig[p]}")
    print("CE4-PILOT-DONE")

if __name__=="__main__": main()
