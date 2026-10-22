"""Bundle the library atlas page with out/viz/viz_data.json into one HTML file.

usage: python review_page/build_atlas.py [viz_data.json] [out.html]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from trabot_a3.paths import OUT  # noqa: E402

data_p = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "viz" / "viz_data.json"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "A3_library_atlas.html"
page = (HERE / "atlas.html").read_text(encoding="utf-8")
data = json.dumps(json.loads(data_p.read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
        "<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>"
        "</head><body>"
        f"<script type=\"application/json\" id=\"atlas-data\">{data}</script>\n{page}</body></html>")
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
