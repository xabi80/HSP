# Presentations: flume mooring and flume wall effect

The `.pptx` files in this folder are **copies** collected on 2026-10-02. Each original stays in
its study folder and is regenerated there; re-copy after a regeneration.

| File | Version | Original, and how it is made |
|---|---|---|
| `Mooring_design_study_rev-D.pptx` | Current: MOORING-SPEC rev D, 2.4 m square platform | `flume-mooring/mooring-design-study/mooring-design-study.pptx` (`flume-mooring/deck_pptx.py mooring-design-study`) |
| `Mooring_briefing_rev-D.pptx` | Current: leadership briefing, rev D | `flume-mooring/leadership-deck/mooring-briefing.pptx` (`flume-mooring/deck_pptx.py leadership-deck`) |
| `Flume_mooring_technical_2026-09-23.pptx` | Earlier: the sizing-phase mooring deck, before the spec | `flume-mooring/Flume_mooring_technical.pptx` (`make_mooring_ppt.py`) |
| `Flume_wall_effect_technical_2.4m-square.pptx` | Current: 2.4 m square platform, 0.49 m to each wall | `platform-12buoy/flume-wall-effect/Flume_wall_effect_technical_sq2p4.pptx` (`PLAT_SQUARE_M=2.4 python make_flume_technical_ppt.py`) |
| `Flume_wall_effect_explained_2.4m-square.pptx` | Current: plain-English, 2.4 m square | `platform-12buoy/flume-wall-effect/Flume_wall_effect_explained_sq2p4.pptx` (`PLAT_SQUARE_M=2.4 python make_flume_ppt.py`) |
| `Flume_wall_effect_technical_2.5m-circle.pptx` | Earlier: 2.5 m circle platform, 0.80 m to each wall | `platform-12buoy/flume-wall-effect/Flume_wall_effect_technical.pptx` (`python make_flume_technical_ppt.py`) |
| `Flume_wall_effect_explained_2.5m-circle.pptx` | Earlier: plain-English, 2.5 m circle | `platform-12buoy/flume-wall-effect/Flume_wall_effect_explained.pptx` (`python make_flume_ppt.py`) |

Notes:
- **The two mooring rev D decks are picture slides.** Each slide is an image of the online deck's
  slide, with its speaker notes. For editable text, use the online deck's
  **Share › Export › PowerPoint**:
  - [Mooring Design Study](https://claude.ai/artifact/9G8QXBHsXMd7kyQyuNZbJe);
  - [Leadership briefing](https://claude.ai/artifact/GiVfP1SYpeHmXdK8PH2VnS).
- **The wall-effect decks are editable** (python-pptx).
- **The heave radiation-damping share differs between versions,** and the difference is the basis,
  not the platform:
  - the 2.4 m decks give about 6.5 % of the platform's FloatSim decay damping;
  - the 2.5 m decks' ~3.5 % is relative to the single hull's ~13 % field-decay damping (see the
    wall-effect README, "Presentations").
