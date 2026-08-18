"""Generate a self-contained technical-report HTML for the fin & array-size study.
Embeds the key figures (down-scaled JPEG data URIs) and builds the result tables from
the fan summary CSVs. Output: studies/platform-16buoy/fin_study/fin_array_study_report.html
"""
from __future__ import annotations

import base64
import csv
from io import BytesIO
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
_FS = _HERE / "fin_study"
_OUT = _FS / "fin_array_study_report.html"
_DATE = "18 August 2026"

_MODELS = [
    ("single buoy", "spar-fin-decay/fin_study", "rao_summary_fin"),
    ("3-cluster", "cluster-3buoy-rigid/fin_study", "rao_summary_cluster_fin"),
    ("12-buoy", "platform-12buoy/fin_study", "rao_summary_platform_fin"),
    ("16-buoy", "platform-16buoy/fin_study", "rao_summary_platform16_fin"),
]
_CONFIGS = [
    ("no fin (+cap)", "none_cap"), ("0.15 m fin · Cd=5", "015_Cd5"),
    ("0.15 m fin · Cd=1", "015_Cd1"), ("0.215 m fin · Cd=5", "0215_Cd5"),
    ("0.215 m fin · Cd=1", "0215_Cd1"),
]
_STUDIES = _HERE.parent  # studies/


def _peaks(folder: str, pfx: str, cfg: str):  # type: ignore[no-untyped-def]
    p = _STUDIES / folder / f"{pfx}{cfg}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    pk = max(rows, key=lambda r: float(r["rao_buoy"]))
    return dict(resT=float(pk["period_s"]), rao=float(pk["rao_buoy"]),
                buoyacc=max(float(r["acc_buoy_amp"]) for r in rows),
                platacc=max(float(r["acc_center_amp"]) for r in rows))


def _img(path: Path, maxw: int = 1500, q: int = 86) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _main_table() -> str:
    out = []
    for clabel, cfg in _CONFIGS:
        out.append(f'<tr class="grp"><td colspan="5">{clabel}</td></tr>')
        for mlabel, folder, pfx in _MODELS:
            d = _peaks(folder, pfx, cfg)
            if d is None:
                continue
            rc = "n hi" if mlabel == "12-buoy" else "n"
            out.append(
                f"<tr><td>{mlabel}</td><td class='n'>{d['resT']:.2f}</td>"
                f"<td class='{rc}'>{d['rao']:.2f}</td>"
                f"<td class='n'>{d['buoyacc']:.2f}</td>"
                f"<td class='n'>{d['platacc']:.2f}</td></tr>")
    return "\n".join(out)


def _fig(uri: str, num: int, cap: str) -> str:
    return (f'<figure><img src="{uri}" alt="Figure {num}"/>'
            f'<figcaption><b>Figure {num}.</b> {cap}</figcaption></figure>')


HTML = r"""<title>Floating Wave Platform — Fin &amp; Array-Size Motion Study</title>
<style>
  :root{
    --bg:#f4f7f8; --card:#ffffff; --ink:#15242c; --ink2:#33474f; --muted:#5e727b;
    --accent:#0c8b96; --accent2:#0a6d76; --warm:#c9772b; --line:#dce5e9;
    --good:#2e8b6f; --warn:#c0563c; --hl:#eef6f7;
    --shadow:0 1px 2px rgba(20,40,50,.06),0 10px 30px rgba(20,40,50,.06);
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","Cascadia Mono","Segoe UI Mono",Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#0c1418; --card:#111e24; --ink:#e7eff2; --ink2:#c2d2d8; --muted:#8ba0aa;
    --accent:#33c3ce; --accent2:#7fe0e6; --warm:#e6974a; --line:#213139;
    --good:#57c39c; --warn:#e6785a; --hl:#132a2c;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 14px 34px rgba(0,0,0,.4);
  }}
  :root[data-theme="dark"]{
    --bg:#0c1418; --card:#111e24; --ink:#e7eff2; --ink2:#c2d2d8; --muted:#8ba0aa;
    --accent:#33c3ce; --accent2:#7fe0e6; --warm:#e6974a; --line:#213139;
    --good:#57c39c; --warn:#e6785a; --hl:#132a2c;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 14px 34px rgba(0,0,0,.4);
  }
  *{box-sizing:border-box}
  html{-webkit-text-size-adjust:100%}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
    line-height:1.62;font-size:16px;padding:clamp(18px,4vw,52px) clamp(14px,4vw,20px)}
  .wrap{max-width:900px;margin:0 auto}
  .eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;
    color:var(--accent);font-weight:600;margin:0 0 8px}
  h1{font-family:var(--serif);font-weight:600;font-size:clamp(1.8rem,4.6vw,2.7rem);line-height:1.1;
    letter-spacing:-.01em;margin:0 0 .35em;text-wrap:balance}
  .dek{font-size:1.06rem;color:var(--ink2);margin:0 0 6px;max-width:66ch}
  .meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:14px;
    border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px 0;
    display:flex;flex-wrap:wrap;gap:5px 22px}
  h2{font-family:var(--serif);font-weight:600;font-size:1.5rem;letter-spacing:-.01em;
    margin:2.6em 0 .1em;padding-top:.5em;border-top:2px solid var(--line)}
  h2 .no{color:var(--accent);font-family:var(--mono);font-size:1.05rem;margin-right:.5em}
  h3{font-size:1.06rem;font-weight:680;margin:1.7em 0 .3em;color:var(--ink)}
  p{margin:.7em 0}
  strong{color:var(--ink);font-weight:670}
  .lead{font-size:1.05rem}
  ul{margin:.6em 0;padding-left:1.2em}
  li{margin:.32em 0}
  a{color:var(--accent2);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--accent2) 40%,transparent)}
  .summary{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
    border-radius:12px;box-shadow:var(--shadow);padding:20px 24px;margin:26px 0}
  .summary h2{border:0;margin:0 0 .5em;padding:0;font-size:1.15rem}
  .summary ol{margin:0;padding-left:1.3em}
  .summary li{margin:.5em 0}
  .callout{background:var(--hl);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:18px 0;font-size:.97rem}
  .callout b{color:var(--accent2)}
  figure{margin:26px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;
    box-shadow:var(--shadow);padding:12px 12px 4px;overflow:hidden}
  figure img{width:100%;height:auto;display:block;border-radius:7px;background:#fff}
  figcaption{font-size:.86rem;color:var(--muted);padding:11px 6px 8px;line-height:1.5}
  figcaption b{color:var(--ink2)}
  .tw{overflow-x:auto;margin:18px 0;border:1px solid var(--line);border-radius:10px}
  table{border-collapse:collapse;width:100%;font-size:.9rem}
  th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
  th{background:var(--card);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
    color:var(--muted);font-weight:700;position:sticky;top:0}
  td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
  td.n.hi{color:var(--warm);font-weight:700}
  tr.grp td{background:color-mix(in srgb,var(--accent) 9%,var(--card));font-weight:700;
    color:var(--accent2);font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}
  .lev{display:grid;grid-template-columns:1fr;gap:12px;margin:18px 0}
  .lev .item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  .lev .item b{display:block;color:var(--accent2);margin-bottom:2px}
  .foot{margin-top:44px;border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:.82rem}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:620px){.grid2{grid-template-columns:1fr}}
  .tag{display:inline-block;font-family:var(--mono);font-size:11px;background:var(--hl);
    border:1px solid var(--line);border-radius:5px;padding:1px 7px;color:var(--ink2)}
</style>
<div class="wrap">
  <p class="eyebrow">FloatSim · Motion &amp; Loads Study</p>
  <h1>Floating Wave Platform — Fin &amp; Array-Size Motion Study</h1>
  <p class="dek">Time-domain simulation of an articulated spar-buoy platform across four array sizes
  and five fin/drag configurations, over a grid of regular sea states. Focus: heave response and
  acceleration, and how they scale with the heave-plate fin, the plate drag, and the number of buoys.</p>
  <div class="meta"><span>%%DATE%%</span><span>1:50 model scale</span>
    <span>single / 3-cluster / 12-buoy / 16-buoy</span><span>FloatSim (Cummins + coupled BEM + KKT joints)</span></div>

  <div class="summary">
    <h2>Executive summary</h2>
    <ol>
      <li><strong>The heave-plate fin is the dominant lever.</strong> The 0.215&nbsp;m fin cuts the peak
      buoy acceleration ~3–4× versus no fin and lengthens the heave natural period from ~2.5&nbsp;s to
      ~3.1–3.4&nbsp;s, moving resonance out of the energetic wave band.</li>
      <li><strong>Plate drag is the largest uncertainty.</strong> Halving the heave-plate drag
      (Cd&nbsp;5&nbsp;→&nbsp;1) roughly <em>doubles</em> the resonant acceleration. Pinning the real
      Cd (tank test) is the key open item.</li>
      <li><strong>Individual-buoy response is non-monotonic in array size:</strong> it rises
      single→3→12 and then <em>drops</em> at 16 buoys (no-fin heave RAO 3.30→3.81→4.27→2.80). The buoys'
      resonant overshoot of the platform is damped out by the denser cluster — confirmed by an
      equivalent-linearized-drag model reproducing the time-domain heave to &lt;0.1%.</li>
      <li><strong>Platform (deck) acceleration is insensitive to array size beyond 12 buoys.</strong>
      It falls monotonically single→12 and then plateaus (no-fin 0.84→0.78→0.70→0.69&nbsp;m/s²). Deck-mounted
      equipment sees the same accelerations at 12 or 16 buoys.</li>
      <li><strong>Lowest accelerations:</strong> the 0.215&nbsp;m fin at Cd=5 — peak buoy accel
      ≈&nbsp;0.22&nbsp;m/s², platform ≈&nbsp;0.29&nbsp;m/s² across the tested sea states.</li>
    </ol>
  </div>

  <h2><span class="no">1</span>Configuration &amp; method</h2>
  <p>The platform is a set of <strong>spar-buoy clusters</strong> — vertical spar floats, each with a
  bottom <strong>heave-plate fin</strong>, connected by rigid arms and yaw-locked articulated joints to a
  central frame — modelled at <strong>1:50 scale</strong>. Four array sizes were run:</p>
  <ul>
    <li><span class="tag">single</span> one isolated spar-buoy;</li>
    <li><span class="tag">3-cluster</span> one cluster of 3 buoys (0.5&nbsp;m radius);</li>
    <li><span class="tag">12-buoy</span> 4 clusters × 3 buoys (square, 1&nbsp;m arm);</li>
    <li><span class="tag">16-buoy</span> 4 clusters × 4 buoys (square, 1&nbsp;m arm).</li>
  </ul>
  <p><strong>Configuration matrix.</strong> Fin diameter {none&nbsp;(+cap), 0.15&nbsp;m, 0.215&nbsp;m} ×
  heave-plate drag {Cd<sub>n</sub>&nbsp;=&nbsp;5, 1} — five configs (the no-fin case carries only a small
  bottom cap). Cd<sub>n</sub>&nbsp;=&nbsp;5 is the textbook heave-plate value; Cd<sub>n</sub>&nbsp;=&nbsp;1
  brackets the low-drag end.</p>
  <p><strong>Sea-state fan.</strong> Regular waves, height <span class="tag">H = 0.04–0.12 m</span> ×
  period <span class="tag">T = 2.0–3.3 s</span> (extended to 3.8&nbsp;s for the 0.215&nbsp;m fin), heading 0°.</p>
  <p><strong>Solver.</strong> FloatSim time-domain Cummins equation with a coupled Capytaine BEM database
  (radiation and diffraction solved across the whole array), velocity-level KKT articulated joints, and
  quadratic Morison drag on the spars and heave plates, after a mandatory static-equilibrium solve and a
  smooth excitation ramp. An independent equivalent-linearized-drag frequency-domain model reproduces the
  time-domain buoy heave to <strong>&lt;0.1%</strong> (Appendix).</p>

  <h2><span class="no">2</span>Cross-model peak response</h2>
  <p>Across every configuration the peak buoy heave response <strong>rises single→3→12 and then falls at
  16 buoys</strong>. Acceleration follows the same shape. The table lists, per model and config, the
  resonance period, the peak buoy heave RAO, and the peak buoy and platform heave accelerations (maxima
  over the H×T fan).</p>
  %%FIG1%%
  <div class="tw"><table>
    <thead><tr><th>model</th><th>res. T (s)</th><th>peak buoy RAO</th><th>peak buoy accel (m/s²)</th>
      <th>peak platform accel (m/s²)</th></tr></thead>
    <tbody>%%TABLE_MAIN%%</tbody>
  </table></div>
  <p class="callout"><b>Read:</b> the 12-buoy column (highlighted) is the worst case for the individual
  buoy in every config; the resonance period lengthens with fin size (~2.5&nbsp;s → ~2.65&nbsp;s →
  ~3.1–3.4&nbsp;s); and dropping the drag from Cd=5 to Cd=1 roughly doubles both RAO and acceleration.</p>

  <h2><span class="no">3</span>Acceleration over the sea-state envelope</h2>
  <p>The 3-D surfaces below show heave acceleration as a function of wave height and period. In every
  config the response forms a <strong>resonance ridge that rises with wave height</strong> and sits at the
  fin-dependent natural period; the ridge marches to longer periods as the fin grows, and stands roughly
  twice as tall at Cd=1 as at Cd=5.</p>

  <h3>3.1 &nbsp;Individual buoy — non-monotonic in array size</h3>
  <p>The representative buoy's acceleration <strong>peaks at 12 buoys and drops at 16</strong> for the
  lightly-damped configs (no fin, 0.15&nbsp;m), and is essentially flat for the heavily-damped 0.215&nbsp;m
  fin. This is the array-size effect: adding the 4th buoy per cluster loads the shared heave mode with more
  drag and damps out the buoys' resonant overshoot.</p>
  %%FIG2%%

  <h3>3.2 &nbsp;Platform centre — monotonic, plateaus by 12</h3>
  <p>The platform/centre acceleration <strong>falls monotonically as the array grows and plateaus by 12
  buoys</strong> (single is the harshest; 12 ≈ 16). More mass and more collective drag smooth the central
  point. This is why array size matters for the individual buoy but not for deck-mounted equipment.</p>
  %%FIG3%%

  <h2><span class="no">4</span>Levers &amp; sensitivities</h2>
  <div class="lev">
    <div class="item"><b>Fin size — primary lever</b>Going none → 0.15&nbsp;m → 0.215&nbsp;m cuts peak buoy
    acceleration from ~0.7–0.9 to ~0.4–0.5 to ~0.22–0.25&nbsp;m/s² (Cd=5), and lengthens the natural period
    ~2.5 → ~2.65 → ~3.1–3.4&nbsp;s, moving resonance away from the wave energy.</div>
    <div class="item"><b>Plate drag Cd — largest uncertainty</b>Cd=1 versus Cd=5 roughly doubles the
    resonant RAO and acceleration (0.15&nbsp;m fin peak buoy accel 1.05 vs 0.47&nbsp;m/s²). The true
    heave-plate Cd is KC- and geometry-dependent and should be measured.</div>
    <div class="item"><b>Array size — matters only for the buoy</b>Individual-buoy response peaks at 12 and
    eases at 16 (no-fin −34% RAO); platform acceleration plateaus by 12. Choose 12 vs 16 on capture, cost
    and layout, not deck loads.</div>
  </div>

  <h2><span class="no">5</span>Conclusions &amp; recommendations</h2>
  <ul>
    <li><strong>Fit the 0.215&nbsp;m heave plate.</strong> It is by far the most effective single change —
    it both minimises accelerations and pushes the natural period out of the energetic 2–3&nbsp;s band.</li>
    <li><strong>Measure the heave-plate drag coefficient.</strong> The Cd=5/Cd=1 spread is a factor-of-two
    on the design acceleration — the biggest open uncertainty. A dedicated forced-oscillation / free-decay
    tank test at representative KC should be the next step.</li>
    <li><strong>Array size (12 vs 16) is not driven by motion.</strong> Both give the same platform
    acceleration; 16 buoys actually lowers the worst individual buoy by ~25–30% in the lightly-damped
    cases. Size the array on power capture, mooring and cost.</li>
    <li><strong>Design point.</strong> 0.215&nbsp;m fin, Cd≈5 → peak buoy accel ≈0.22&nbsp;m/s², platform
    ≈0.29&nbsp;m/s² over H≤0.12&nbsp;m — the quietest configuration tested.</li>
  </ul>

  <h2><span class="no">6</span>Method notes, validation &amp; caveats</h2>
  <p>The frequency-domain <strong>equivalent-linearized-drag</strong> model — the exact FloatSim drag,
  harmonically linearized and folded into the impedance — reproduces the nonlinear time-domain buoy heave
  to &lt;0.1% (below), confirming the results are drag-limited and correctly captured.</p>
  %%FIGVAL%%
  <p><strong>Interactive 3-D playback</strong> of the real simulated motion (drag to orbit, scrub the wave
  cycle) is available for the
  <a href="https://claude.ai/code/artifact/bddf6911-8ef8-4774-8d21-c168fc89171c">12-buoy</a> and
  <a href="https://claude.ai/code/artifact/7517ed97-4d6d-48bb-8abb-892f064a79d3">16-buoy</a> platforms,
  sharing the same five fin/H/T cases.</p>
  <p class="callout"><b>Caveats.</b> 1:50 model scale; regular (monochromatic) waves, not irregular sea
  states; calm-water Morison drag (wave-orbital velocity in the drag term is a planned refinement); single
  wave heading (0°). Absolute accelerations are model-scale; Froude-scale to full size for design use.</p>

  <div class="foot">FloatSim time-domain campaign · figures and tables generated from the committed fan
  summaries (<span class="tag">rao_summary_*.csv</span>) · %%DATE%% · internal engineering report.</div>
</div>
"""


def main() -> None:
    fig1 = _fig(_img(_FS / "fin_single_vs_cluster_vs_12_vs_16.png", 1400),
                1, "Peak buoy heave RAO (left) and Nz acceleration (right) versus model size, per fin "
                   "(Cd=5). Both rise single→3→12 then fall at 16 buoys.")
    fig2 = _fig(_img(_FS / "accel_buoy_HT_surf3d_4models.png", 1600),
                2, "Representative-buoy heave acceleration (m/s²) over wave height H and period T. Columns: "
                   "single / 3-cluster / 12-buoy / 16-buoy; rows: fin/Cd config. Peak (●) rises to 12 buoys "
                   "then drops at 16 for the lightly-damped configs.")
    fig3 = _fig(_img(_FS / "accel_HT_surf3d_4models.png", 1600),
                3, "Platform/centre heave acceleration (m/s²) over H and T, same layout. Falls "
                   "monotonically with array size and plateaus by 12 buoys.")
    figval = _fig(_img(_FS / "drag_fd_validation.png", 1100),
                  4, "Validation: the equivalent-linearized-drag frequency-domain model (teal) reproduces "
                     "the time-domain fan (red) buoy RAO to <0.1% for both 12- and 16-buoy platforms; the "
                     "radiation-only model (grey) is ~15–20× too high — the response is drag-limited.")
    html = (HTML.replace("%%DATE%%", _DATE)
                .replace("%%TABLE_MAIN%%", _main_table())
                .replace("%%FIG1%%", fig1).replace("%%FIG2%%", fig2)
                .replace("%%FIG3%%", fig3).replace("%%FIGVAL%%", figval))
    _OUT.write_text(html, encoding="utf-8")
    print(f"wrote {_OUT}  ({_OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
