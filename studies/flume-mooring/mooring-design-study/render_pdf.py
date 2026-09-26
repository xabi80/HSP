"""Render the technical deck's source (deck.json + slides/*.html, the files of the online Slides
deck) to mooring-design-study.pdf and SPEAKER-NOTES.md, with headless Edge (as build_spec.py).

The online deck is the master: re-read its files into this folder after editing it there, then
run this script. The deck's images are uploaded assets: the project's own figures (../figs, from
mooring_figures.py) with their white margins trimmed. IMAGES maps each asset id to its figure;
trim() reproduces the trimmed copy. The PDF approximates the online deck's rendering (same slide
HTML, a small stylesheet for the Slides runtime's defaults); the PowerPoint file comes from the
online deck's Share > Export.

Run: python render_pdf.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PDF_NAME = "mooring-design-study.pdf"
IMAGES = {
    "f9b13b27d48c3256917416e8af7dee83": "plan_buoy",
    "5b03f69303836a658ffc3c0610e87641": "plan_cluster",
    "41e85135e04fb4e195ea0a4a86f1050e": "plan_platform",
    "c300f2bff091d9c42c968b028f8028f5": "elevation_buoy",
    "39de03f0967508f6b5b14defec4eb151": "elevation_cluster",
    "57f7c82e8e4f93625d331e542757c65f": "elevation_platform",
    "6cda2eacb540ac124962af15f1d948f7": "rationale",
    "a63bacae17ffeebd086e9c488c237b14": "offsets",
}
CSS = """@page{size:1920px 1080px;margin:0}html,body{margin:0;padding:0}
section{box-sizing:border-box;width:1920px;height:1080px;position:relative;overflow:hidden;
break-after:page;page-break-after:always}
h1,h2,h3,p,ul,ol,table{margin:0}h1{line-height:1.1}h2{line-height:1.15}h3{line-height:1.2}
p{line-height:1.4}ul,ol{padding-left:1.1em}li{margin:0 0 6px}img{display:block}
table{border-collapse:collapse}th,td{padding:0.35em 0.6em;text-align:left;vertical-align:top;
border-bottom:1px solid #C9D4D8}th{font-weight:600;border-bottom:2px solid #9FB3BF}
aside{display:none}"""


def trim(name: str, pad: int = 16) -> bytes:
    """The figure with its white margins cropped to `pad` px, as uploaded to the deck."""
    im = Image.open(FIGS / f"{name}.png").convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255)))
    x0, y0, x1, y1 = diff.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox()
    box = (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
    buf = io.BytesIO()
    im.crop(box).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def main() -> None:
    deck = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))
    links = "".join(
        f'<link rel="stylesheet" href="{html.escape(f["href"])}">'
        for f in deck["faces"].values()
        if "href" in f
    )
    uris = {
        aid: "data:image/png;base64," + base64.b64encode(trim(name)).decode()
        for aid, name in IMAGES.items()
    }
    slides, notes = [], [f"# {deck['title']}: speaker notes", ""]
    for i, sid in enumerate(deck["order"], start=1):
        s = (HERE / "slides" / f"{sid}.html").read_text(encoding="utf-8")
        for aid, uri in uris.items():
            s = s.replace(f"/_blob/{aid}", uri)
        m = re.search(r"<aside>(.*?)</aside>", s, re.S)
        title = re.search(r"<h[12][^>]*>(.*?)</h[12]>", s, re.S)
        notes += [
            f"## {i}. {html.unescape(title.group(1)) if title else sid}",
            "",
            html.unescape(m.group(1).strip()) if m else "(no notes)",
            "",
        ]
        slides.append(s)
    head = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(deck['title'])}"
        f"</title>{links}<style>{CSS}</style></head><body>"
    )
    page = head + "\n".join(slides) + "</body></html>"
    (HERE / "SPEAKER-NOTES.md").write_text("\n".join(notes), encoding="utf-8")
    htm, pdf = HERE / "_render.html", HERE / PDF_NAME
    htm.write_text(page, encoding="utf-8")
    pdf.unlink(missing_ok=True)
    subprocess.run(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf}",
            htm.as_uri(),
        ],
        check=False,
        timeout=180,
        capture_output=True,
    )
    last = -1  # Edge may finish writing after it returns
    for _ in range(120):
        size = pdf.stat().st_size if pdf.exists() else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(1.0)
    htm.unlink(missing_ok=True)
    print(f"wrote {pdf.name} and SPEAKER-NOTES.md" if pdf.exists() else "PDF not produced")


if __name__ == "__main__":
    main()
