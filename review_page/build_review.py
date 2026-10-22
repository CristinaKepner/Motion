"""Bundle the review page and its data into one HTML file that opens from disk.

usage: python review_page/build_review.py [bundle.json] [out.html]  → review_page/dist/A3_motion_review.html
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from trabot_a3.paths import OUT  # noqa: E402
bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "library" / "review_bundle.json"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "dist" / "A3_motion_review.html"

page = (HERE / "src" / "review.html").read_text(encoding="utf-8")
data = json.dumps(json.loads(bundle.read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))
data = data.replace("</", "<\\/")   # never close the <script> early
html = ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
        "<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>"
        "</head><body>"
        f"<script type=\"application/json\" id=\"bundle-data\">{data}</script>\n"
        f"{page}</body></html>")
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
