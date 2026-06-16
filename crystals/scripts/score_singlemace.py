import sys
sys.path[:0]=["/home/ubuntu/code/py/dielectric","/home/ubuntu/packages/lemat-genbench/src"]
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from chem.stability import compute_e_above_hull, load_stability_calc
from lemat_genbench.metrics.sun_metric import SUNMetric
ats=read(sys.argv[1],index=":")
calc=load_stability_calc(device="cuda"); sun=SUNMetric()
structs=[]
for a in ats:
    try: eah=compute_e_above_hull(a,calc=calc,timeout=120).get("e_above_hull")
    except Exception: eah=None
    s=AseAtomsAdaptor.get_structure(a)
    if eah is not None: s.properties["e_above_hull_mean"]=float(eah)
    structs.append(s)
m=sun.compute(structs).metrics
print(f"SINGLE-MACE: n={len(structs)} stable={m.get('stable_count')} meta={m.get('metastable_count')} "
      f"SUN={m.get('sun_rate'):.3f} MSUN={m.get('msun_rate'):.3f} Combined={m.get('combined_sun_msun_rate'):.3f}")
