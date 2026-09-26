# Mooring Design Study: speaker notes

## 1. Mooring Design Study

This deck walks through the flume mooring design at the depth needed to review it: the acceptance criteria, how FloatSim was used to choose the attachment height and the two cord sets, what the verification runs show, the loads and hardware, the installation checks, and the limits of the analysis. Every number comes from the study's JSON records and from MOORING-SPEC rev C.1, which is generated from the same records. The three configurations are shown to scale on their own slides.

## 2. Specification complete; interference and excursion met

Three messages. The specification is complete and waits only on HWRL confirming seven facility assumptions, the first being that anchors can be fixed to the flume walls under water. The mooring does not distort the measurement: the tilt natural period moves by at most 1.37 percent, inside each article's zeta/3 tolerance, and the heave natural period by at most 0.5 percent. And it keeps the articles in the tracking window even at H = 0.5 m. Both are possible because the rig uses two cord sets on the same anchors and attachment points: a soft set for the response tests and a stiff set for the load and survival tests.

## 3. Three articles, natural periods inside the wave band

Three articles are tested in turn in the OSU Large Wave Flume at 1:50. The table gives the unmoored heave and tilt natural periods from FloatSim's modal analysis: heave 2.56 to 2.61 s, tilt 2.76 to 2.92 s. Both sit inside the wave band, so the response tests excite them directly, and the mooring must not move them. These periods are from the flume BEM databases, whose 12-sided waterline under-reads the displaced volume, and so the heave and pitch restoring, by 4.5 percent; every resonance here is about 0.04 s (heave) to 0.08 s (tilt) long. Re-meshed until converged, the single buoy's heave period is 2.519 s and its pitch period 2.684 s, which matches the separate fine-mesh OSU buoy model, 2.52 and 2.69 s, and the unofficial field heave-decay video, about 2.50 s. The mooring's effect is unaffected, because moored and free are computed on the same database and move together. The regeneration at 36 sides, which corrects all three articles, is planned before Phase D. The matrix has an operational band for the motions and an extreme band for loads and offsets.

## 4. Station-keeping and non-interference conflict

The core conflict, written as pass/fail numbers before any design. Station-keeping wants stiff lines: the article must stay inside the tracking window, clear of the walls, with taut lines in the response tests and a threefold margin on loads. Non-interference wants soft lines: the mooring must not move the tilt or heave natural periods, and its own slow modes must sit at least four times above the longest wave period. The tilt limit comes from each article's damping: a period shift no larger than a third of zeta keeps the resonant amplitude within about 5 percent, with zeta from FloatSim sweeps at H = 0.04 m, where it is smallest: 4.25, 3.56 and 3.15 percent.

## 5. Criteria first, every option simulated in FloatSim

The method. The criteria were fixed first, so the design could not move the goalposts. Every option was a FloatSim deck, evaluated the same way: a static settle for the calm tilt and at-rest tensions, a modal analysis for the natural periods, rigid pulls for the stiffness, and regular-wave runs. The drift force is deliberately conservative: the recorded fixed-body bound is applied in-run at each spar's waterline, on top of FloatSim's own mean force. A written rule picked the design, the chosen designs were verified in wave runs, and the specification is generated from the same records. FloatSim runs at LEVEL1, valid to 0.1 rad; near resonance the articles tilt 10 to 30 degrees, so those predictions are indicative.

## 6. Four revisions, each correcting the last

The design changed in the open. Rev A attached at the pin plane to remove calm tilt; its tilt stiffening had not been compared with the resonance bandwidth, and the comparison showed 6.5 to 8.6 times the tolerance, so it was withdrawn. Rev B swept the attachment height and chose low, submerged attachments; it meets the interference limit, but its rule omitted the confirmed excursion criterion, and H = 0.5 m drifts the cluster and platform 2.5 to 3.4 m. Rev C keeps rev B as the operational set and adds a stiffer extreme set on the same anchors. Rev C.1 adds the cord creep allowance. The commits are a91e1d5, fb6ba5b, 4f8a2f8 and 73a516d.

## 7. Attach below the waterline: coupling ≈ k·h²

The key physical insight. The pin plane was chosen to remove the calm tilt: the pretension moment acts about the pins. But the dynamic coupling between the lines and the tilt mode acts about the mode's rotation centre, near the centre of gravity, and scales roughly as k times h squared, h being the attachment's height above that centre. At the pin, h is large and the tilt period shortened by about 9 percent. The sweep in the figure ran six heights at two stiffnesses; filled markers pass every hard criterion, the stars are the chosen designs. Attached 0.15 m below the waterline (cluster, platform) or 0.50 m (buoy), the shift falls inside zeta/3. The price for the cluster and platform is a 1 degree calm tilt from the pretension, which the installers use to check the tension. The buoy's collar balances its lines, so it has no calm tilt.

## 8. Single buoy

The single buoy, to scale: plan on top, elevation looking across the flume below. Four lines run from a slender radial collar of 0.2 m radius, 0.50 m below still water, to wall anchors at the same depth, so the lines run level; the inset shows the 21 mm catenary sag. The collar balances the four lines, so there is no calm tilt, and the pretension is checked by measuring line tension. The mooring moves the heave natural period by 0.5 percent and the tilt period by 1.37 percent, just inside the 1.42 percent limit. Those periods are from the flume BEM, whose 12-sided waterline makes them long; on a converged mesh the free buoy's heave period is 2.519 s and its pitch period 2.684 s, which matches the OSU buoy model and the field heave-decay video, about 2.50 s. The shifts are unaffected. The buoy has one cord set: its predicted H = 0.5 m offsets stay within 0.68 m. Its yaw is stiff, 1.12 s, because the collar turns pretension into yaw stiffness; that is the source of its extreme-wave stability issue.

## 9. Cluster

The cluster, to scale. In plan, each spar takes one line to a wall anchor, an X-spread 20.1 degrees off the flume axis; the shaded box is the tracking window, and the cluster keeps 1.39 m from each wall. In elevation, the lines attach 0.15 m below still water and run level to anchors at the same depth. The operational cord is 4.15 N/m; the extreme cord is five times stiffer at the same at-rest tension, 1.44 N, so both sets give the same 1 degree calm tilt. The heave natural period moves by 0.08 percent, the tilt period by 0.73 percent against a 1.19 percent limit.

## 10. 4×4 platform

The 4 by 4 platform, to scale. Four two-leg bridles, eight legs, run from the up- and down-stream row spars to the wall anchors, two legs per anchor. The platform is 2.06 m wide at 45 degrees and keeps 0.80 m from each wall. The legs attach 0.15 m below still water. The operational legs are 8.32 N/m; the extreme legs six times stiffer, at the same 1.42 N at rest. The heave natural period is unchanged to 0.03 percent; the tilt period moves 0.72 percent against a 1.05 percent limit. The platform's up-flume anchors carry the largest working load in the study, 267.4 N, which sets the 300 N anchor rating.

## 11. The mooring barely moves the natural periods

The first result: the mooring does not distort the measurement. The natural periods barely move: tilt by at most 1.37 percent, each inside its zeta/3 limit, heave by at most 0.5 percent, a seventh of the heave tolerance because heave is far more damped, zeta about 11 percent. The third column comes from 48 FloatSim runs, moored and free in the same waves: at the tilt resonance, which for the buoy is also near its heave response peak, the heave RAO changes by -0.2 to +4.6 percent. The biggest heave change in those runs, +9 percent, is at 2.2 s, on the flank of the response and inside the flume's sloshing exclusion window, so it is not a test period; it is not an effect on the heave resonance. Criterion 6 asks for 3 percent on the heave RAO, and some of these values exceed it, so the formal check, with the numerically restrained free reference, is carried to Phase D. The tilt peaks differ by about the 1 degree calm tilt.

## 12. The extreme set keeps H = 0.5 m in view

The second result: station-keeping. At H = 0.5 m, the two worst periods, with the conservative drift sum, the soft operational cords let the cluster and platform settle 2.5 to 3.4 m down-flume, far outside the tracking window that ends at +1.0 m. The stiff extreme set, five and six times the operational stiffness, holds the maximum surge, mean plus wave motion, at 0.97 and 0.95 m. The figure shows it to scale: grey is the calm position, red the operational set's mean positions, teal the extreme set's. In the response tests the operational cords hold the mean offset under 7 cm with every line taut. The single buoy needs only one set.

## 13. One anchor layout, two cord sets

Operationally the rig is simple: the anchors and attachment points never move. The operational cords are used for every response test; for the load and survival tests they are swapped for cords five or six times stiffer, tensioned to the same at-rest tension. Matching the at-rest tension, rather than the nominal pretension, makes the calm line forces identical, so the operational calm equilibrium holds for both sets. The extreme cords' pre-stretch is only a few centimetres, so they are set by tension or by the calm tilt, never by length. Their surge period, 10.5 and 9.6 s, is below the 14 s separation criterion; that is declared and accepted, because these tests measure loads. Since both sets look the same at rest, a static pull confirms which is on; the thresholds are the geometric means of the two sets' stiffness.

## 14. Key decisions and their technical basis

Five decisions worth knowing. We evaluated removing the operational pretension: the wave-frequency response barely changes, but the lines would go slack for up to 40 percent of each resonant cycle, which FloatSim's quasi-static line does not model, so the tank and the model would stop seeing the same mooring; the pretension was kept. The buoy keeps its pretension for a different reason: its collar turns it into yaw stiffness. One compromise cord cannot meet both the separation and the excursion criteria at H = 0.5 m, hence two sets. The extreme set was sized on the stricter reading of the excursion criterion. And the two sets share the at-rest tension, not the nominal pretension, so their calm equilibrium is identical.

## 15. Anchors, cords, collar and the daily re-check

The hardware is modest. Four wall anchors per article, under water at the attachment depth, each rated at 300 N or more; the largest working load anywhere, with the factor of three, is 267.4 N on the platform's up-flume anchors with the extreme set. Elastic cord usable under water, in two stiffnesses; the extreme cord is bought as a minimum stiffness because a softer one would leave the window. The buoy's collar must be slender, because it is a submerged appendage the model does not include; a disk would add about 21 kg of water mass. And rubber creeps: about 13 percent of the tension is lost in a day, so the tension or the 1 degree calm tilt is checked before each day's first run, with re-tension thresholds. The platform's operational set and the buoy are re-tensioned before their most sensitive cases.

## 16. Risks and limitations

What could still change the design. The HWRL facility assumptions come first: underwater wall anchors and their rating in particular; any no means a redesign, so nothing is bought until they are confirmed. FloatSim runs at LEVEL1, valid to 5.7 degrees, and the resonant cases tilt 10 to 33 degrees, so those results, loads included, are indicative until the LEVEL2 rerun. The single buoy's extremes are partly unpredicted: three cases diverge in FloatSim and six grow yaw slowly; they run last and stop at any yaw growth. The heave-RAO form of the interference criterion is not yet demonstrated to 3 percent in every case. Snap loads when a slack line re-tensions are not simulated. And the BEM regeneration at 36 waterline sides is pending.

## 17. From specification to the flume

Four steps from specification to the flume. The HWRL confirmations gate everything. Then procurement of both cord sets and the anchors, confirming the cord's creep rate. In parallel, the verification: the BEM regeneration at 36 waterline sides and the full Phase D matrix, 442 cases across both cord sets, which includes the formal heave-RAO check of criterion 6. And the LEVEL2 rerun of the resonant cases, which also decides whether the single buoy's divergence in the extremes is physical.

## 18. Where every number comes from

Every number in this deck can be traced to a record in the study folder. The specification and the figures are generated from these JSON files, and the design basis records each decision and its evidence. The moored-versus-free comparison comes from the 48 runs behind the 3D motion viewer. A longer, 27-slide version of this deck is in the repository history at commit fd633a8.
