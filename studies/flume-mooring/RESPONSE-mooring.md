# Response to reviewer comment — mooring for the flume tests

**Reviewer comment:** *which mooring is needed, and how could the flume size influence it?*

## Summary

The flume mooring's only job is **station-keeping**. It holds each article in the test section
against the mean wave-drift force and must not change the wave-frequency response we measure
(free decay, heave and pitch response, accelerations). The field site is deep water, so a
2.7 m-deep flume cannot host a scale model of the field mooring. We therefore use an
**equivalent soft horizontal mooring**. We characterise it by a static pull test and include it
in the numerical model of the test.

**The mooring:** four soft lines in an X-spread, **horizontal at the still-water line (SWL)**,
running to wall anchors 5 m up- and downstream. Each line is a linear spring in series with
low-stretch rope. They are sized for a **surge natural period of ~15 s**, nearly four times the
longest test-wave period (4 s).

We checked the design by adding the mooring to our coupled multibody models of all three
articles. The models include the full inter-buoy hydrodynamics and the pinned joints. With the
mooring connected:

- heave natural periods shift by **< 0.5 %**;
- pitch periods of the buoy, cluster hub and platform deck shift by **< 1 %**;
- the pinned buoys' tilt mode shifts by **≤ 2.4 %**;
- the mean trim is **zero**.

All of this is within normal model-test uncertainty. **Decay tests can be run with the mooring
connected.**

## Per article

**1 buoy (Phase 1).**
- All four lines attach to one collar on the spar at the SWL.
- Per line: 2.0 N/m, 2.2 N pretension, 4.1 N maximum tension.
- Mean offset under the conservative H = 0.5 m drift: 0.74 m (0.27 m at H = 0.3 m).
- Heave period −0.43 %, pitch period −0.10 %, no trim.
- Attaching at the spar top instead would give −1.3 % pitch period and a 0.8° mean trim.

**1 cluster, 4 buoys (Phase 2, 45° test orientation).**
- The cluster sits square to the flume, with two buoys facing the waves. Each of the four lines
  attaches to the nearest spar, at the SWL.
- Per line: 8.3 N/m, 9.0 N pretension, 16.5 N maximum tension.
- Mean offset: 0.71 m (0.27 m at H = 0.3 m).
- Heave −0.45 %, hub pitch −0.45 %, buoy-tilt mode −2.4 %, no hub trim, and no buoy tilt: each
  spar holds its own drift.

**4×4 platform (Phase 3, 45° test orientation).**
- The lines attach at the SWL to the upstream and downstream rows of four spars. Each line ends
  in a 2-leg bridle to two spars.
- Per line: 33 N/m, 36 N pretension, 66 N maximum tension.
- Mean offset: 0.75 m (0.28 m at H = 0.3 m).
- Heave −0.45 %, deck pitch −0.36 %, buoy-tilt mode −2.3 %, no deck trim.

The three designs give the same surge period and the same offset because the horizontal inertia
and the drift both scale with the number of buoys. The design scales by buoy count: per-line
stiffness and pretension are ×4.1 for the cluster and ×16.6 for the platform.

## Why the lines attach at the waterline

The mean drift on these slender spars acts in the splash zone. Lines at the SWL react it with no
lever arm. For the articulated cluster and platform we compared three alternatives.

- **A single bridle at the deck / pin level** (+0.72 m).
  - It causes no trim.
  - But it holds the light deck nearly still. That stiffens the mode in which the buoys swing
    together about their pins (2.86–2.92 s, inside the wave band) by ~9 %.
  - To bring that below 3.5 %, the mooring would need to be softer than 25 s, and the mean offset
    would then exceed 2 m.
- **Only the four corner buoys.**
  - The whole platform's drift passes through four pins, tilting those buoys 9°.
- **Below the waterline.**
  - The drift then tilts the attached buoys up to 11°.

With the recommended attachment, the platform's largest buoy tilt under drift is 3.0° at
H = 0.5 m (1.1° at H = 0.3 m); the cluster's buoys do not tilt at all. This tilt is not a
mooring effect. It comes from the drift acting 0.72 m below each
pin, and appears with any station-keeping, in the flume or in the field.

## How the flume size influences the mooring

- **Depth (2.7 m)** prevents a scale model of the deep-water field mooring, which is why the
  flume mooring is an equivalent one.
  - It also rules out floor anchors on short lines. From the SWL to a floor anchor 5 m away, a
    line is inclined about 27°. That would add vertical stiffness (~30 % of the surge stiffness) and a
    downward pull on the article. Horizontal lines to wall anchors avoid both.
  - Finite depth increases the drift of long waves (×1.9 at 4 s compared with deep water). But
    the design drift is set by the steepest waves near 2.2 s, where depth adds only 5 %. **Depth
    does not drive the design.**
- **Width (3.66 m)** sets the X-spread angle (20°), and so the sideways stiffness: about ⅓ of
  surge, a 26 s sway period.
  - A lateral disturbance of 10 % of the drift moves the platform 0.21 m. It has 0.80 m of
    clearance to each wall (1.39 m for the cluster, 1.69 m for the buoy).
  - Longer anchor spans would soften the small heave coupling further. But they would roughly
    double the lateral excursion toward the walls. **The flume width is what fixes the anchors
    at about ±5 m.**
- **Length (104 m)** is not a constraint. Offsets below 0.8 m and a ±5.3 m line footprint fit
  easily; wave gauges are referenced to the mean moored position.

## Conservatism and verification

- The design drift is an **upper bound** for spars held fixed: no shielding between spars, and
  the steepest waves of the test matrix.
- We also computed the drift on the **moving** articles, using drag-limited time-domain
  simulations with the flow velocity relative to each spar's waterline.
  - Away from the buoys' own resonances, the articles move with the water and the drift is a
    small fraction of the bound: ≤ 0.1 N (buoy), ≤ 0.9 N (cluster) and ≤ 11 N (platform),
    against 5 / 21 / 84 N.
  - Near the buoys' pitch and tilt resonances the mean drift reverses and points upstream, and
    can approach the bound's magnitude. Those cases involve large buoy tilts, beyond the model's
    small-angle range.
  - The mooring is therefore sized on the bound in both directions, which the symmetric
    X-spread provides.
- Before testing, a static pull test will verify the mooring stiffness and pretension. Load
  cells on the lines will measure the actual mean drift during the tests, which is itself useful
  data.
- The measured mooring will be included in the numerical model used to interpret the tests.
