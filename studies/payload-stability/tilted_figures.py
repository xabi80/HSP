"""Pictures of the mass-shift cases (tilted_cases.json) for the tilt-in-waves deck: the payload
centred, then shifted to 5 deg and 10 deg, drawn to scale from the static equilibria with the
payload decks' drawing code. The cluster (payload_figures.draw) has 12.5 kg on its hub and the
platform (platform_figures.draw) 50 kg on its deck, each shifted along +x, the wave direction.
Every picture uses its deck's fixed view and top-view inset.

Writes wseq_cluster.png and wseq_platform.png. Run: python tilted_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import payload_figures as pfc
import platform_figures as pfp
import platform_statics as ps

HERE = Path(__file__).resolve().parent
NOTE = {"centred": "payload centred", "tilt5": "shifted: 5° tilt", "tilt10": "shifted: 10° tilt"}


def main() -> None:
    cases = json.loads((HERE / "tilted_cases.json").read_text(encoding="utf-8"))["cases"]
    for art, width in (("cluster", 3.1), ("platform", 3.3)):
        sel = [c for c in cases if c["article"] == art and c["case"] in NOTE]
        fig, axs = plt.subplots(1, len(sel), figsize=(width * len(sel), 4.7))
        for ax, c in zip(axs, sel, strict=True):
            q = np.asarray(c["q"])
            if art == "cluster":
                e = np.array([c["offset_m"], 0.0, 0.0])
                pfc.draw(ax, q, c["m_kg"], e, note=NOTE[c["case"]])
            else:
                ld = ps.payload(c["m_kg"], c["offset_m"], 0.0, 0.0)
                pfp.draw(ax, q, ld, note=NOTE[c["case"]])
        fig.tight_layout()
        fig.savefig(HERE / f"wseq_{art}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print("wrote wseq_cluster.png, wseq_platform.png")


if __name__ == "__main__":
    main()
