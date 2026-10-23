"""Build the single-file hand library / agent page.

usage: python review_page/build_hand.py → review_page/dist/A3_hand_library.html
inputs: results/hand_library/{library.json, poses_sheet.png, style_curves.png}
        results/hand_demo/{timeline.json, stream_log.jsonl, contact_sheet.png, seek_5s.png, seek_5s.json}
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trabot_a3.paths import OUT  # noqa: E402

HERE = Path(__file__).resolve().parent
LIB, DEMO = OUT / "hand_library", OUT / "hand_demo"


def jpeg(p: Path, max_w: int = 2400, q: int = 80) -> str:
    im = Image.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    lib = json.loads((LIB / "library.json").read_text())
    tl = json.loads((DEMO / "timeline.json").read_text())
    rows = [json.loads(l) for l in (DEMO / "stream_log.jsonl").read_text().splitlines() if l.strip()]
    stream = {"t": [r["t"] for r in rows], "L": [r["L_o10"] for r in rows], "R": [r["R_o10"] for r in rows],
              "act": [[r["active"]["L"], r["active"]["R"]] for r in rows]}
    for a in lib["atoms"]:   # trim bulky per-atom fields the page does not show
        a["annotation"].pop("fingertips_palm_m", None)
    data = {"lib": lib, "timeline": tl, "stream": stream, "seek": json.loads((DEMO / "seek_5s.json").read_text()),
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "img": {"seek": jpeg(DEMO / "seek_5s.png", 1200), "sheet": jpeg(DEMO / "contact_sheet.png"),
                    "poses": jpeg(LIB / "poses_sheet.png"), "curves": jpeg(LIB / "style_curves.png", 1400, 85)}}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (HERE / "src" / "hand.html").read_text()
    out = HERE / "dist" / "A3_hand_library.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>A3 灵巧手动作库</title>"
                   "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"></head><body>"
                   f"<script type=\"application/json\" id=\"hand-data\">{blob}</script>\n{page}</body></html>")
    print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
