"""Coupled check of the flume station-keeping mooring on the three test articles, using the
real FloatSim models: the single OSU buoy (6 DOF), the Phase-2 cluster (4 buoys + hub, 30 DOF,
yaw_locked KKT joints) and the Phase-3 4x4 platform (21 bodies, 126 DOF, 45 deg test
orientation = axis-aligned 4x4 grid).

The mooring is the X-spread from mooring_sizing.py at the design point (T_surge = 15 s, wall
anchors 5 m up/downstream): an ANISOTROPIC point stiffness diag(Kx, Ky, Kz) at the attachment
(Kz is only the geometric T0/l of the pretensioned horizontal lines -- the deck LinearSpring is
isotropic and would wrongly add Kx to heave), plus a nominal bridle yaw stiffness. It is added
to the system stiffness as B^T K B with B = [I, -skew(r)] for an attachment offset r.

Attachment options are the physically realisable ones: on the single buoy, the spar at the
waterline / top (+0.72 m) / CoG depth (-0.91 m); on the pinned articles, a single-point bridle at
the pin plane (hub / deck frame, +0.72 m -- every pin in the model sits there) or the lines
split by bridles over the bow and stern spars (cluster: bow + stern buoy; platform: the 4-buoy
bow and stern rows, or only the 4 corner buoys) at the waterline or at CoG depth.

For each article and attachment:
  * NATURAL PERIODS (surge, heave, pitch), moored vs unmoored: constrained generalized eigen-
    problem on the null space of the joint Jacobian, iterated on the frequency-dependent added
    mass -- the decay-test quantities the mooring must not disturb. Each mode is identified in
    the unmoored model (heave: rigid-heave MAC; deck pitch: largest deck-pitch participation;
    buoy tilt: the in-phase pendulum mode of the pinned buoys) and tracked into the moored model
    by mass-weighted MAC; moored surge is the best rigid-surge MAC.
  * STATIC EQUILIBRIUM under the mean drift (splash-zone force at every spar's waterline, the
    conservative H = 0.5 m and H = 0.3 m values from mooring_sizing.py): mean surge offset,
    deck/hub trim, and the largest buoy tilt about its pin (articulated models).

Also sweeps the design surge period (10-30 s) for the two finalist attachments of each pinned
article (pin plane; bow/stern spars at the waterline) and the buoy's waterline collar: the softer
the mooring, the smaller its influence on the dynamics and the larger the offset.

Writes mooring_verify.csv, mooring_tsurge_sweep.csv, mooring_verify.png.
Run: python mooring_verify.py
"""
# ruff: noqa: E402, E702, RUF001  -- sys.path bootstrap first; compact lines; minus sign in labels
from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PLAT_ROT_DEG", "45")   # platform at the 45 deg test orientation

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FWE = HERE.parent / "platform-12buoy" / "flume-wall-effect"
for p in (ROOT, FWE, HERE.parent / "osu-test-buoy", HERE):
    sys.path.insert(0, str(p))
warnings.simplefilter("ignore")

import articulated_wall as aw
import mooring_sizing as ms
import osu_buoy_common as obc

from floatsim.driver import build_system
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    PlateMember,
    Simulation,
    YawLockedJoint,
    distributed_cylinder_drag,
)
from floatsim.io.deck import RegularWave as DeckWave

T_SURGE, L_ANCHOR, R_BRIDLE = 15.0, 5.0, 0.30
H_DESIGN = (0.5, 0.3)


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def point_k(n, body, r, kxyz, kyaw=0.0):
    """Global stiffness of an anisotropic point spring at body-frame offset r (+ yaw spring)."""
    B = np.hstack([np.eye(3), -skew(np.asarray(r, float))])
    K6 = B.T @ np.diag(kxyz) @ B
    K6[5, 5] += kyaw
    K = np.zeros((n, n)); K[6 * body:6 * body + 6, 6 * body:6 * body + 6] = K6
    return K


def mooring_k(n, points, kxyz, kyaw):
    """X-spread split equally over the attachment points; the bridle yaw spring only applies to a
    single-point attachment (split points get their yaw stiffness from Ky x lever)."""
    share = 1.0 / len(points)
    K = np.zeros((n, n))
    for body, r in points:
        K += point_k(n, body, r, [share * k for k in kxyz], kyaw if len(points) == 1 else 0.0)
    return K


def gen_force(n, body, r, fx):
    """Generalized force of a horizontal force fx (+x) at body-frame offset r."""
    f = np.zeros(n); f[6 * body] = fx
    f[6 * body + 4] = r[2] * fx; f[6 * body + 5] = -r[1] * fx
    return f


def _afun(hdb, hydro_dof, n):
    om = np.asarray(hdb.omega); A = np.asarray(hdb.A); hd = np.asarray(hydro_dof)

    def f(w):
        i = int(np.clip(np.searchsorted(om, w), 1, len(om) - 1))
        t = (w - om[i - 1]) / (om[i] - om[i - 1]); t = min(max(t, 0.0), 1.0)
        Ag = np.zeros((n, n)); Ag[np.ix_(hd, hd)] = (1 - t) * A[:, :, i - 1] + t * A[:, :, i]
        return Ag
    return f


# ------------------------------------------------------------------ the three articles
def article_buoy():
    hdb = obc.load_hdb(); lhs = obc.build_lhs(hdb); n = 6
    M = np.asarray(lhs.M_plus_Ainf) - np.asarray(hdb.A_inf)
    return dict(name="1 buoy", n=n, M=M, C=np.asarray(lhs.C), A=_afun(hdb, range(6), n), G=None,
                ref=0, buoys=[0], drift=[(0, (0.0, 0.0, 0.0))], nspar=1,
                pos=np.zeros((1, 3)),
                att={"spar top (+0.72 m)": [(0, (0, 0, 0.718))],
                     "spar, waterline": [(0, (0, 0, 0))],
                     "spar, CoG depth (-0.91 m)": [(0, (0, 0, -0.907))]})


def _cluster_deck():
    spar = distributed_cylinder_drag(z_bottom=aw._SPAR_BOT_B, z_top=aw._WL_B, diameter=aw.SPAR_D,
                                     cd=aw.SPAR_CD, n_segments=10)
    plate = PlateMember(type="plate", center=[0.0, 0.0, aw._PLATE_B], normal=[0.0, 0.0, 1.0],
                        radius=aw.PLATE_R, thickness=aw.PLATE_T, Cd_n=aw.PLATE_CDN,
                        Cd_t=aw.PLATE_CDT)
    r = 0.5 * 1.25 / 1.5; bodies: list = []; joints: list = []
    for k, a in enumerate(np.deg2rad([0.0, 90.0, 180.0, 270.0])):
        bx, by = r * np.cos(a), r * np.sin(a)
        bodies.append(Body(name=f"buoy{k + 1}", reference_point=[bx, by, aw.ZB], mass=aw.M_BUOY,
                           inertia=Inertia(Ixx=aw.IXX, Iyy=aw.IYY, Izz=aw.IZZ),
                           hydro_body_label=f"buoy{k + 1}", initial_conditions=InitialConditions(),
                           drag_elements=[*spar, plate]))
        joints.append(YawLockedJoint(type="yaw_locked", body_a=f"buoy{k + 1}", body_b="hub",
                                     attach_a_body=[0.0, 0.0, aw.ZH - aw.ZB],
                                     attach_b_body=[bx, by, 0.0], axis=[0, 0, 1.0]))
    bodies.append(Body(name="hub", reference_point=[0.0, 0.0, aw.ZH], mass=2.0,
                       inertia=Inertia(Ixx=0.3, Iyy=0.3, Izz=0.6), structural=True))
    return Deck(simulation=Simulation(duration=10.0, dt=0.01),
                environment=Environment(water_depth=200.0, water_density=aw.RHO, gravity=aw.G),
                waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
                bodies=bodies, joints=joints,
                shared_hydro_database=HydroDatabaseRef(format="capytaine", path="placeholder.nc"),
                output=Output(file="o.h5", channels=["heave"], sample_rate=10.0))


def _articulated(name, dk, nc, ref, z_ref, spar_sets):
    hdb = read_capytaine(nc)
    setup = build_system(dk, bem_databases={}, dt=0.01, t_max_kernel=30.0, solve_equilibrium=True,
                         shared_hydro_database=hdb, asymptote_check_override=aw.OVR,
                         kernel_decay_floor_override=aw.OVR)
    hd = aw.hydro_dof(dk); n = setup.lhs.M_plus_Ainf.shape[0]
    Ainf = np.zeros((n, n)); Ainf[np.ix_(hd, hd)] = np.asarray(hdb.A_inf)
    buoys = [k for k, b in enumerate(dk.bodies) if b.hydro_body_label is not None]
    return dict(name=name, n=n, M=np.asarray(setup.lhs.M_plus_Ainf) - Ainf,
                C=np.asarray(setup.lhs.C), A=_afun(hdb, hd, n),
                G=np.asarray(setup.constraints.jacobian(setup.xi0)), ref=ref, buoys=buoys,
                drift=[(k, (0.0, 0.0, -aw.ZB)) for k in buoys], nspar=len(buoys),
                pos=np.array([b.reference_point for b in dk.bodies], float),
                att=_spar_attachments(ref, z_ref, spar_sets))


def _spar_attachments(ref, z_ref, spar_sets):
    att = {"pin plane (+0.72 m)": [(ref, (0, 0, aw.ZH - z_ref))]}
    for label, ks, depths in spar_sets:
        if "waterline" in depths:
            att[f"{label}, waterline"] = [(k, (0, 0, -aw.ZB)) for k in ks]
        if "cog" in depths:
            att[f"{label}, CoG depth (-0.91 m)"] = [(k, (0, 0, 0)) for k in ks]
    return att


def _outer_buoys(dk, which):
    """Bridle attachment buoys: 'rows' = every buoy of the bow and stern rows (extreme |x|;
    the bow + stern buoy for the cluster), 'corners' = the 4 grid corners."""
    pos = [(k, b.reference_point) for k, b in enumerate(dk.bodies) if b.hydro_body_label]
    xs = np.array([p[0] for _, p in pos]); ys = np.array([p[1] for _, p in pos])
    if which == "corners":
        return [k for (k, p) in pos
                if np.isclose(abs(p[0]), xs.max()) and np.isclose(abs(p[1]), ys.max())]
    return [k for (k, p) in pos if np.isclose(abs(p[0]), xs.max())]


def article_cluster():
    dk = _cluster_deck()
    return _articulated("1 cluster (4 buoys)", dk, HERE / "cluster_osu_open_psd.nc",
                        len(dk.bodies) - 1, aw.ZH,
                        [("bow+stern spars", _outer_buoys(dk, "rows"), ("waterline", "cog"))])


def article_platform():
    dk = aw.deck()
    return _articulated("4x4 platform (45°)", dk, FWE / f"coupled_osu_open{aw.SUF}_psd.nc",
                        aw.PLAT, aw.ZP,
                        [("bow+stern rows (8 spars)", _outer_buoys(dk, "rows"),
                          ("waterline", "cog")),
                         ("4 corner spars", _outer_buoys(dk, "corners"), ("waterline",))])


# ------------------------------------------------------------------ analyses
def null_space(G, n):
    return np.eye(n) if G is None else sla.null_space(G)


W_FLOOR = (2 * np.pi / 40.0) ** 2   # modes slower than 40 s are the regularised rigid modes


def rigid_pattern(art, dof):
    """Rigid-assembly pattern: 'surge' / 'heave' translation of every body, or 'tilt' = every
    buoy pitching in phase (the pendulum mode of pinned buoys)."""
    u = np.zeros(art["n"])
    for k in range(len(art["pos"])):
        if dof == "surge":
            u[6 * k] = 1.0
        elif dof == "heave":
            u[6 * k + 2] = 1.0
        elif k in art["buoys"]:
            u[6 * k + 4] = 1.0
    return u


def roll_split(n, eps=1e-2):
    """Negligible roll spring on every body: splits the pitch/roll degenerate pairs of the
    symmetric articles so eigenvectors come out as pure pitch or pure roll (period effect ~1e-5)."""
    K = np.zeros((n, n))
    for k in range(n // 6):
        K[6 * k + 3, 6 * k + 3] = eps
    return K


def _eig(art, K, w):
    N = null_space(art["G"], art["n"]); Mw = art["M"] + art["A"](w)
    lam, V = sla.eig(N.T @ K @ N, N.T @ Mw @ N); lam = lam.real
    phi = N @ V.real
    phi = phi / np.sqrt(np.abs(np.einsum("ij,ij->j", phi, Mw @ phi)))   # M-normalised
    return lam, phi, Mw


def by_mac(u):
    """Mode with the largest mass-weighted MAC against pattern / mode shape u."""
    def pick(phi, Mw, lam):
        mac = (phi.T @ Mw @ u) ** 2 / (u @ Mw @ u)
        mac[~(lam > W_FLOOR * 0.25)] = 0.0
        j = int(np.argmax(mac)); return j, float(mac[j])
    return pick


def by_dof(g):
    """Mode with the largest participation of generalized coordinate g (M-normalised shapes)."""
    def pick(phi, Mw, lam):
        part = phi[g] ** 2; part[~(lam > W_FLOOR)] = 0.0
        j = int(np.argmax(part)); return j, float(part[j])
    return pick


def track(art, K, pick, T0):
    """Constrained generalized eigenproblem on the null space of the joint Jacobian, iterated on
    A(omega); returns (period, mode shape, pick score) of the mode `pick` selects."""
    w = 2 * np.pi / T0
    for _ in range(40):
        lam, phi, _Mw = _eig(art, K, w)
        j, score = pick(phi, _Mw, lam); w_new = float(np.sqrt(lam[j]))
        if abs(w_new - w) < 1e-7 * w:
            break
        w = w_new
    return 2 * np.pi / w_new, phi[:, j], score


def static(art, K, F):
    n = art["n"]; G = art["G"]
    if G is None:
        return np.linalg.solve(K, F)
    m = G.shape[0]
    KK = np.zeros((n + m, n + m)); KK[:n, :n] = K; KK[:n, n:] = G.T; KK[n:, :n] = G
    return np.linalg.solve(KK, np.concatenate([F, np.zeros(m)]))[:n]


T_SWEEP = (10.0, 15.0, 20.0, 25.0, 30.0)
SWEEP_ATT = {"1 buoy": ["spar, waterline"],
             "1 cluster (4 buoys)": ["pin plane (+0.72 m)", "bow+stern spars, waterline"],
             "4x4 platform (45°)": ["pin plane (+0.72 m)", "bow+stern rows (8 spars), waterline"]}


def mooring_design(art, t_surge, f_spar):
    """X-spread stiffness for a target surge period: Kx = M_h (2 pi / T)^2 (sizing-script M_h),
    Ky and the geometric Kz from mooring_sizing.xspread, nominal bridle yaw."""
    a_ms = ms.ARTICLES[art["name"]]
    Kx = (a_ms["M"] + a_ms["A"]) * (2 * np.pi / t_surge) ** 2
    xs = ms.xspread(Kx, art["nspar"] * f_spar / Kx, 0.25, L_ANCHOR)
    return (Kx, xs["Ky"], xs["Kz"]), Kx * R_BRIDLE ** 2


def reference_modes(art):
    """Unmoored modes (tiny surge/sway/yaw regularisation keeps the rigid null modes below
    W_FLOOR): deck pitch = largest deck-pitch participation (for the buoy: its own pitch);
    buoy tilt = the in-phase pendulum mode of the pinned buoys."""
    n, r = art["n"], art["ref"]
    K0 = art["C"] + roll_split(n) + point_k(n, r, (0.0, 0.0, 0.0), (1e-6, 1e-6, 0.0), 1e-6)
    ref = {"pitch": track(art, K0, by_dof(6 * r + 4), 2.3)}
    if art["G"] is not None:
        ref["tilt"] = track(art, K0, by_mac(rigid_pattern(art, "tilt")), 2.9)
    return ref


def evaluate(art, points, kxyz, kyaw, ref_modes, env):
    n, r = art["n"], art["ref"]
    K = art["C"] + roll_split(n) + mooring_k(n, points, kxyz, kyaw)
    T_s, _, m_s = track(art, K, by_mac(rigid_pattern(art, "surge")), T_SURGE)
    # heave: Rayleigh quotient of the rigid-heave pattern (what a heave decay test excites;
    # robust to the mooring mixing the platform's near-degenerate heave-like modes)
    uh = rigid_pattern(art, "heave")
    out = dict(Kx_Npm=round(kxyz[0], 2), Ky_Npm=round(kxyz[1], 2), Kz_geo_Npm=round(kxyz[2], 2),
               T_surge_s=round(T_s, 2), surge_MAC=round(m_s, 3),
               dT_heave_pct=round(100 * (np.sqrt((uh @ art["C"] @ uh) / (uh @ K @ uh)) - 1), 3))
    for key in ("pitch", "tilt"):
        if key not in ref_modes:
            out.update({f"T_{key}_unmoored_s": "", f"T_{key}_moored_s": "",
                        f"dT_{key}_pct": "", f"{key}_MAC": ""})
            continue
        T0_, phi0, _ = ref_modes[key]
        T1_, _, mac_ = track(art, K, by_mac(phi0), T0_)
        out.update({f"T_{key}_unmoored_s": round(T0_, 4), f"T_{key}_moored_s": round(T1_, 4),
                    f"dT_{key}_pct": round(100 * (T1_ / T0_ - 1), 3),
                    f"{key}_MAC": round(mac_, 3)})
    for H in H_DESIGN:
        F = np.zeros(n)
        for (k, rw) in art["drift"]:
            F += gen_force(n, k, rw, env[H])
        xi = static(art, K, F)
        tilt = max(abs(np.degrees(xi[6 * k + 4])) for k in art["buoys"])
        out[f"offset_H{H}_m"] = round(float(xi[6 * r]), 3)
        out[f"trim_H{H}_deg"] = round(float(np.degrees(xi[6 * r + 4])), 3)
        out[f"buoy_tilt_H{H}_deg"] = round(float(tilt), 3)
    return out


def _write(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    env = {H: max(sum(ms.drift_per_spar(H, T)[:2]) for T in ms.T_WAVE) for H in H_DESIGN}
    rows, sweep = [], []
    for build in (article_buoy, article_cluster, article_platform):
        art = build(); name = art["name"]
        ref_modes = reference_modes(art)
        kxyz, kyaw = mooring_design(art, T_SURGE, env[0.5])
        print(f"\n=== {name}: {art['n']} DOF | Kx {kxyz[0]:.1f}, Ky {kxyz[1]:.1f}, "
              f"Kz(geo) {kxyz[2]:.2f} N/m, yaw {kyaw:.2f} N·m/rad | unmoored "
              + ", ".join(f"{k} {v[0]:.3f} s" for k, v in ref_modes.items()), flush=True)
        for att_name, points in art["att"].items():
            out = {"article": name, "attachment": att_name,
                   **evaluate(art, points, kxyz, kyaw, ref_modes, env)}
            rows.append(out)
            tl = f"{out['dT_tilt_pct']:+6.2f}%" if out["dT_tilt_pct"] != "" else "   n/a"
            print(f"  [{att_name:>36}] T_surge {out['T_surge_s']:5.1f} s | heave "
                  f"{out['dT_heave_pct']:+5.2f}% | pitch {out['dT_pitch_pct']:+6.2f}% | tilt {tl}"
                  f" | H0.5: offset {out['offset_H0.5_m']:.2f} m, "
                  f"trim {out['trim_H0.5_deg']:+.2f}°, "
                  f"buoy tilt {out['buoy_tilt_H0.5_deg']:.2f}°", flush=True)
        for att, ts in [(a_, t_) for a_ in SWEEP_ATT[name] for t_ in T_SWEEP]:
            kxyz_, kyaw_ = mooring_design(art, ts, env[0.5])
            out = {"article": name, "attachment": att, "T_surge_design_s": ts,
                   **evaluate(art, art["att"][att], kxyz_, kyaw_, ref_modes, env)}
            sweep.append(out)
            tl = f"{out['dT_tilt_pct']:+6.2f}%" if out["dT_tilt_pct"] != "" else "   n/a"
            print(f"  sweep T_surge {ts:4.0f} s [{att}]: pitch {out['dT_pitch_pct']:+6.2f}% | "
                  f"tilt {tl} | offset H0.5 {out['offset_H0.5_m']:.2f} m, "
                  f"H0.3 {out['offset_H0.3_m']:.2f} m", flush=True)
    _write("mooring_verify.csv", rows); _write("mooring_tsurge_sweep.csv", sweep)
    plot(rows, sweep)
    print("\nwrote mooring_verify.csv, mooring_tsurge_sweep.csv, mooring_verify.png")


def _att_style(att):
    if "top" in att or "pin plane" in att:
        return "#b2432c", "above water, +0.72 m (spar top / pin plane)"
    if "corner" in att:
        return "#8fb9bb", "waterline, 4 corner spars only"
    if "waterline" in att:
        return "#0c8b96", "waterline (buoy spar / bow+stern spars)"
    return "#e08214", "CoG depth, −0.91 m (spar)"


def plot(rows, sweep):
    arts = list(dict.fromkeys(r_["article"] for r_ in rows))
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.6)); ax = ax.ravel()
    w = 0.2; seen = set()
    for i, a in enumerate(arts):
        sel = [r_ for r_ in rows if r_["article"] == a]
        for j, s_ in enumerate(sel):
            xx = i + (j - (len(sel) - 1) / 2) * w
            col, lab = _att_style(s_["attachment"])
            lab = lab if lab not in seen else None; seen.add(lab)
            ax[0].bar(xx, s_["dT_pitch_pct"], w, color=col, label=lab)
            if s_["dT_tilt_pct"] != "":
                ax[1].bar(xx, s_["dT_tilt_pct"], w, color=col)
            ax[2].bar(xx, max(abs(s_["trim_H0.5_deg"]), s_["buoy_tilt_H0.5_deg"]), w, color=col)
    ax[1].text(0, -0.4, "n/a\n(no pins)", ha="center", va="top", fontsize=9, color="0.35")
    titles = ["(a) Pitch period shift of the measured body (%)\n"
              "buoy: own pitch · cluster: hub · platform: deck",
              "(b) Buoy-tilt (pendulum) mode period shift (%)\n"
              "pinned buoys swinging in phase about their pins",
              "(c) Largest mean buoy tilt under drift (deg)\nH = 0.5 m upper-bound drift"]
    for a_, t_ in zip(ax[:3], titles, strict=True):
        a_.set_xticks(range(len(arts))); a_.set_xticklabels(arts, fontsize=9)
        a_.set_title(t_, fontsize=10.5, fontweight="bold")
        a_.axhline(0, color="0.4", lw=0.8); a_.grid(axis="y", alpha=0.3)
        a_.set_xlim(-0.5, len(arts) - 0.5)
    ax[0].legend(title="lines attached at", fontsize=8.5, title_fontsize=8.5, loc="lower right")
    # (d) softness sweep
    a4 = ax[3]; a4b = a4.twinx()
    for a, att in [(a_, t_) for a_ in arts for t_ in SWEEP_ATT[a_]]:
        ss = [s_ for s_ in sweep if s_["article"] == a and s_["attachment"] == att]
        ts = [s_["T_surge_design_s"] for s_ in ss]
        key = "dT_tilt_pct" if ss[0]["dT_tilt_pct"] != "" else "dT_pitch_pct"
        col, _ = _att_style(att)
        mk = {"1 buoy": "o", "1 cluster (4 buoys)": "s", "4x4 platform (45°)": "D"}[a]
        what = "tilt mode" if key == "dT_tilt_pct" else "pitch"
        a4.plot(ts, [s_[key] for s_ in ss], marker=mk, color=col, lw=1.8, ms=6,
                ls="-" if "platform" in a or a == "1 buoy" else "--",
                label=f"{a.split(' (')[0]}: {what}, {att.split(' (')[0].split(',')[0]}")
    ss = [s_ for s_ in sweep if s_["article"] == arts[-1] and "pin plane" in s_["attachment"]]
    for H, ls in zip(H_DESIGN, ["-", ":"], strict=True):
        a4b.plot([s_["T_surge_design_s"] for s_ in ss], [s_[f"offset_H{H}_m"] for s_ in ss], ls,
                 color="0.55", lw=2.2, alpha=0.6, label=f"mean offset, H = {H} m (all articles)")
    a4.set_xlabel("design surge period T_surge (s)"); a4.set_ylabel("period shift (%)")
    a4b.set_ylabel("mean surge offset (m)", color="0.35")
    a4.axhline(0, color="0.4", lw=0.8); a4.grid(alpha=0.3)
    a4.set_title("(d) Softer mooring: less dynamic influence, more offset", fontsize=10.5,
                 fontweight="bold")
    h1, l1 = a4.get_legend_handles_labels(); h2, l2 = a4b.get_legend_handles_labels()
    a4.set_ylim(-19.0, 1.0); a4b.set_ylim(0.0, 3.2)
    a4.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), frameon=False)
    hv = sorted({round(float(r_["dT_heave_pct"]), 2) for r_ in rows})
    fig.suptitle(f"Flume station-keeping mooring on the three articles — coupled FloatSim models "
                 f"(X-spread, T_surge = {T_SURGE:.0f} s, anchors ±{L_ANCHOR:.0f} m)\n"
                 f"heave natural period {hv[0]:+.2f} to {hv[-1]:+.2f} % for every "
                 f"article/attachment (geometric stiffness of the pretensioned lines)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "mooring_verify.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
