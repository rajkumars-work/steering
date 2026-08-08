import sys
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from lemat_genbench.preprocess.multi_mlip_preprocess import create_multi_mlip_preprocessor
from lemat_genbench.metrics.sun_metric import SUNMetric
ats=read(sys.argv[1],index=":")
structs=[AseAtomsAdaptor.get_structure(a) for a in ats]
pre=create_multi_mlip_preprocessor(
    mlip_names=["orb","mace","uma"], relax_structures=False, n_jobs=1, extract_embeddings=False,
    mlip_configs={"orb":{"model_type":"orb_v3_conservative_inf_mpa","device":"cuda"},  # _mpa, not _omat -- see README "orb calibration"
                  "mace":{"model_type":"mp","device":"cuda"},
                  "uma":{"model_name":"uma-s-1p1","task":"omat","device":"cuda"}})
res=pre.run(structs); sun=SUNMetric()
m=sun.compute(res.processed_structures).metrics
print(f"3MLIP(orb+mace+uma): n={len(res.processed_structures)} stable={m.get('stable_count')} "
      f"meta={m.get('metastable_count')} SUN={m.get('sun_rate'):.3f} MSUN={m.get('msun_rate'):.3f} "
      f"Combined={m.get('combined_sun_msun_rate'):.3f}", flush=True)
