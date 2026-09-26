"""Render the leadership deck's source (deck.json + slides/*.html, the files of the online Slides
deck) to leadership-briefing.pdf and SPEAKER-NOTES.md, with headless Edge (as build_spec.py).

The online deck is the master: re-read its files into this folder after editing it there, then
run this script. The images the deck holds as uploaded assets are the project's own figures
(../figs); IMAGES maps each asset id to its file. The PDF approximates the online deck's
rendering (same slide HTML, a small stylesheet for the Slides runtime's defaults); the
PowerPoint file comes from the online deck's Share > Export.

Run: python render_pdf.py
"""
from __future__ import annotations

import base64
import html
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
IMAGES = {"811a0730779f4cec52f3275383605a7a": HERE.parent / "figs" / "plan_platform.png",
          "c12cf90b5d58c7d918eb1c0525bcb3f4": HERE.parent / "figs" / "elevation_cluster.png"}
CSS = """@page{size:1920px 1080px;margin:0}html,body{margin:0;padding:0}
section{box-sizing:border-box;width:1920px;height:1080px;position:relative;overflow:hidden;
break-after:page;page-break-after:always}
h1,h2,h3,p,ul,ol,table{margin:0}h1{line-height:1.1}h2{line-height:1.15}h3{line-height:1.2}
p{line-height:1.4}ul{padding-left:1.1em}li{margin:0 0 6px}img{display:block}
table{border-collapse:collapse}th,td{padding:0.35em 0.6em;text-align:left;vertical-align:top;
border-bottom:1px solid #C9D4D8}th{font-weight:600;border-bottom:2px solid #9FB3BF}
aside{display:none}"""


def main() -> None:
    deck = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))
    links = "".join(f'<link rel="stylesheet" href="{html.escape(f["href"])}">'
                    for f in deck["faces"].values() if "href" in f)
    slides, notes = [], [f"# {deck['title']}: speaker notes", ""]
    for i, sid in enumerate(deck["order"], start=1):
        s = (HERE / "slides" / f"{sid}.html").read_text(encoding="utf-8")
        for aid, path in IMAGES.items():
            uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
            s = s.replace(f"/_blob/{aid}", uri)
        m = re.search(r"<aside>(.*?)</aside>", s, re.S)
        title = re.search(r"<h[12][^>]*>(.*?)</h[12]>", s, re.S)
        notes += [f"## {i}. {html.unescape(title.group(1)) if title else sid}", "",
                  html.unescape(m.group(1).strip()) if m else "(no notes)", ""]
        slides.append(s)
    head = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(deck['title'])}"
            f"</title>{links}<style>{CSS}</style></head><body>")
    page = head + "\n".join(slides) + "</body></html>"
    (HERE / "SPEAKER-NOTES.md").write_text("\n".join(notes), encoding="utf-8")
    htm, pdf = HERE / "_render.html", HERE / "leadership-briefing.pdf"
    htm.write_text(page, encoding="utf-8")
    pdf.unlink(missing_ok=True)
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf}", htm.as_uri()], check=False, timeout=180,
                   capture_output=True)
    last = -1                       # Edge may finish writing after it returns
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
