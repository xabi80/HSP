# Flume Mooring Design — Leadership Briefing: speaker notes

## 1. Mooring the wave-flume test articles

This briefing covers why the test articles need a mooring, how we designed it, what the simulations show it achieves, the decisions we made, and what is still open before we buy hardware. All numbers come from our FloatSim simulations and the mooring specification, rev C.1.

## 2. The bottom line

Three messages. First, the design is finished and documented; the only thing between us and buying hardware is the test lab confirming seven assumptions about their facility, such as whether we can fix anchors to the flume walls under water. Second, the mooring does not distort the measurement: it changes the buoys' natural rocking period by at most 1.4 percent, which is inside the limit we agreed before designing. Third, it keeps the articles where the cameras can see them, even in the biggest waves. The trick that makes both possible is two sets of cords on the same anchors.

## 3. Three test articles, one wave flume

We are testing three scale models in Oregon State's Large Wave Flume: a single buoy, a four-buoy cluster and a sixteen-buoy platform. The purpose of the tests is to measure how they move in waves so we can check our simulator, FloatSim, against real data. That purpose drives everything about the mooring: it must not become the thing we end up measuring.

## 4. The mooring has two jobs that pull apart

The core tension. To hold an article in the camera's view and off the walls, you want stiff lines. To avoid changing the motion you are measuring, you want soft lines. We wrote both goals down as pass or fail numbers before designing anything. The rocking-period limit, 1.0 to 1.4 percent depending on the article, comes from how sharply each article resonates: a shift that small changes the peak response by no more than about 5 percent.

## 5. How we designed it

The method matters as much as the answer. We fixed the acceptance criteria before designing, so nobody could move the goalposts. Every option was simulated in FloatSim rather than estimated by hand. A written decision rule chose the design. The chosen design was then run through full wave simulations. And the specification tables and figures are generated from the same simulation files, so the documents cannot disagree with the analysis.

## 6. Four revisions, each fixing the last one

The design went through four revisions in the open. Revision A attached the cords above the water to avoid tilting the buoys, but the simulation showed it stiffened their rocking by about 9 percent, six to nine times our tolerance, so it was withdrawn. Revision B moved the attachment just under the water with low tension; it measures true, but in the biggest waves the soft cords let the cluster and platform drift three and a half metres. Revision C keeps revision B for measuring and adds a stiffer cord set for the big-wave tests. Revision C.1 adds a daily tension check because rubber cords relax over time.

## 7. Elastic cords to underwater wall anchors

This is the arrangement for the largest article, the 16-buoy platform, drawn to scale from the specification. Four soft elastic cords run from the article's up- and down-stream rows to anchors on the flume walls, five metres away. Everything sits just under the water: the cords attach 15 centimetres below the water line, and the anchors are at the same depth so the cords run level. The single buoy uses a slender collar 50 centimetres down. The platform keeps 0.8 metres of clearance to each wall.

## 8. Why the cords attach below the water line

The single most important design insight. Where a cord attaches decides how much it interferes. Attached high, at the hinge line above the water, the cord acts like a lever against the buoys' rocking and shifted the rocking period by about 9 percent. Attached just below the water line, close to the point the buoys rock about, the lever almost disappears and the shift drops under 1 percent. The price is a small, steady lean of one degree caused by the cord tension. We turned that into a feature: measuring the lean is how the installers check the tension.

## 9. The mooring barely changes the motion

The first result: the mooring does not distort what we measure. The biggest change to the natural rocking period is 1.4 percent, on the single buoy; the cluster and platform are at 0.7 percent, each inside its own tolerance. We also ran every article in the simulator with and without the mooring, in the same waves. Where the buoys move the most, near their rocking resonance and in long waves, the heave response agrees within 5 percent; the largest difference, 9 percent, is at one intermediate wave period. The one visible effect is a steady one-degree lean from the cord tension, which leadership had already confirmed as acceptable.

## 10. Stiff cords keep big-wave tests in view

The second result: position. The chart shows the furthest the cluster and platform move down the flume in the biggest waves we plan to run, half a metre high, including a deliberately conservative drift force. With the soft measurement cords they drift three and a half metres, far outside the camera window, which ends one metre down-flume. With the stiff cords they stay at 0.95 to 0.97 metres, just inside. In the smaller waves used for the measurements, the soft cords already hold them within 7 centimetres. The single buoy needs only one set: its offsets stay inside the window with its normal cords.

## 11. One rig, two cord sets

Operationally this is simple. The anchors and attachment points never move. For the measurement series we fit the soft cords; for the big-wave load and survival series we swap to cords five to six times stiffer, tensioned to the same resting tension. Because both sets sit at the same one-degree lean at rest, you cannot tell them apart by looking, so the procedure requires a quick pull test before every big-wave series: the stiff set resists five to six times harder. Running big waves on the soft cords by mistake would drift the article about three and a half metres and overload those cords.

## 12. Key decisions and why

Four decisions worth knowing. We looked hard at removing the cord tension altogether; the motion would barely change, but the cords would go slack for up to 40 percent of every wave at resonance, and a flapping cord is something our simulator cannot reproduce, so the tank and the model would stop seeing the same mooring. We kept the tension. The single buoy keeps its tension for a different reason: its collar turns that tension into resistance against spinning. We chose two cord sets rather than one compromise cord, because no single stiffness satisfies both jobs. And we sized the stiff set against the stricter reading of the camera-window limit.

## 13. What we will buy

The hardware is modest. Wall anchors under water, four per article, each rated at 300 newtons or more; the largest working load anywhere is 267 newtons, on the platform in the biggest waves, and that already includes a threefold safety margin. Elastic shock cord suitable for use under water, in two stiffnesses. For the single buoy, a slender collar: a thin rod or wire spreader, because a solid disk would drag about 21 kilograms of water with it and change the very motion we measure. And a daily routine: rubber cords relax by around 13 percent over a day, so the tension is checked before each day's first run.

## 14. What could still change the design

Four things could still change the design. First, the test lab has not yet confirmed our facility assumptions, most importantly that anchors can be fixed to the walls under water; any "no" means a redesign, which is why nothing is bought until they confirm. Second, our simulator uses a small-angle approximation, and near resonance the buoys tilt beyond it; those results are indicative, and an upgrade is planned. Third, in three of the largest-wave cases the simulator cannot predict the single buoy, which starts to spin; we run those last, build wave height gradually and stop at any sign of spin. Fourth, the snap load when a slack cord re-tensions is not simulated; loads carry a threefold margin and the installation includes a pull test with a load cell.

## 15. From specification to the flume

Four steps take us from specification to the flume. The first gates everything: the test lab must confirm our seven facility assumptions before we buy anything. Then procurement of the cords and anchors, confirming the cord's creep rate with the supplier. In parallel we finish the verification: a refined model of the buoys and the full 442-case simulation matrix. And before the resonant, large-tilt cases are analysed, the simulator needs its planned upgrade for large rotations. Replace the bracketed line with the decision you are asking for.

## 16. Supporting material

For anyone who wants the detail: the full specification, the design basis with every decision and its evidence, the evaluation of whether the cords need tension, and an interactive 3D viewer that plays the simulated motion of each article with and without the mooring. All are generated from the same simulation files as the numbers in this briefing.
