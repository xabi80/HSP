"""Drag-FD decomposition across fins {none,015,0215} x {12,16} at each measured peak,
to explain why the 12->16 reversal is BIGGEST for no-fin. Tests whether the reversal
tracks the buoy/platform ARTICULATION overshoot (which the fin suppresses)."""
import sys, warnings
from pathlib import Path
import numpy as np
_R = Path("studies")
for p in ("platform-12buoy","platform-16buoy"): sys.path.insert(0,str(_R/p))
sys.path.insert(0,"floatsim"); warnings.simplefilter("ignore")
from drag_fd import solve, harmonic_drag
from floatsim.hydro.excitation import interpolate_rao

FINS=[("none",0.0841),("015",0.15),("0215",0.215)]
PEAKT={(12,"none"):2.50,(12,"015"):2.65,(12,"0215"):3.30,
       (16,"none"):2.50,(16,"015"):2.80,(16,"0215"):3.40}
MEAS={(12,"none"):4.27,(12,"015"):2.37,(12,"0215"):1.84,
      (16,"none"):2.80,(16,"015"):1.89,(16,"0215"):1.58}

def build(model,fin,plate_r):
    if model==12:
        import platform_fin_fan as pff, platform_rao_pilot as prp; N=12
    else:
        import platform16_fin_fan as pff, platform16_rao as prp; N=16
    hdb=pff._hdb(fin); setup=pff._build(plate_r,5.0,hdb)
    hydro=np.asarray(prp._hydro_dof(prp._deck_with_drag()))
    return hdb,setup,hydro,[6*prp._buoy_body_index(k)+2 for k in range(N)],6*prp._buoy_body_index_platform()+2

amp=0.02; res={}
print(f"{'fin':6} {'model':>5} {'T':>5} {'buoyRAO':>8} {'meas':>6} {'platRAO':>8} {'buoy/plat':>9}",flush=True)
for fin,pr in FINS:
    for model in (12,16):
        T=PEAKT[(model,fin)]; om=2*np.pi/T
        hdb,setup,hydro,bhv,phv=build(model,fin,pr)
        xd,_=solve(hdb,setup,hydro,om,amp,bhv,phv,drag=True)
        b=np.abs(xd[bhv[6]])/amp; p=np.abs(xd[phv])/amp
        res[(model,fin)]=(b,p,b/p)
        print(f"{fin:6} {model:5d} {T:5.2f} {b:8.3f} {MEAS[(model,fin)]:6.2f} {p:8.3f} {b/p:9.2f}",flush=True)
print("\n== reversal vs 12-buoy articulation overshoot ==",flush=True)
print(f"{'fin':6} {'RAO12/16':>9} {'overshoot12(buoy/plat)':>22} {'overshoot16':>12}",flush=True)
for fin,_ in FINS:
    b12=res[(12,fin)][0]; b16=res[(16,fin)][0]
    print(f"{fin:6} {b12/b16:9.2f} {res[(12,fin)][2]:22.2f} {res[(16,fin)][2]:12.2f}",flush=True)
print("DONE",flush=True)
