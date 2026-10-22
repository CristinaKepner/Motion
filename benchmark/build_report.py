"""Assemble datasets.json + benchmark results into one HTML report page.

usage: python benchmark/build_report.py  → review_page/A3_gesture_benchmark.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trabot_a3.paths import OUT  # noqa: E402

HERE = Path(__file__).resolve().parent
B = OUT / "benchmark"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ds = json.loads((HERE / "datasets.json").read_text())
    lib = load(B / "library_report.json")
    results = []
    if lib:
        for s in lib["sources"]:
            results.append(s)
    for name, key in (("Bandai Namco 手势子集（真实动捕，97 段）", "report_bandai.json"), ("合成演讲数据（流程验证，含音频）", "report_beat.json")):
        r = load(B / key)
        if r:
            r["name"] = name
            results.append(r)
    data = {"datasets": ds, "results": results, "compare": lib["compare"] if lib else {}, "self_checks": lib["self_checks"] if lib else {},
            "adapter": load(B / "adapter_selfcheck.json"), "protocol": (HERE / "PROTOCOL.md").read_text()}
    page = (HERE.parent / "review_page" / "benchmark.html").read_text()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out = HERE.parent / "review_page" / "A3_gesture_benchmark.html"
    out.write_text("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
                   "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
                   "<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style></head><body>"
                   f"<script type=\"application/json\" id=\"bench-data\">{blob}</script>\n{page}</body></html>")
    print("wrote", out, f"{out.stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
