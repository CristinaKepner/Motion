"""Sourcing (provenance) tree for the motion library.

library
 └─ dataset            (where the motion came from: real retargeted mocap / sim-generated)
     └─ recording      (one source file, with its detection statistics b, Δ, σ, θ)
         ├─ rejected   (segments the offline pipeline dropped, with the reason)
         └─ atom       (an approved-candidate Motion Atom, with repairs and quality)
             └─ variant (a stylised / mirrored child generated in simulation)

Every node carries a rollup over the clips beneath it — count, mean score, grade mix,
share of C/D, flag histogram — so a reviewer can start at the top and walk down to
whichever source or style is producing unnatural or incoherent clips.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _rollup(scores: list[float], grades: list[str], flags: list[list[str]], n_rej: int = 0,
            n_atoms: int = 0, n_variants: int = 0) -> dict:
    g = Counter(grades)
    fl = Counter(f for fs in flags for f in fs)
    n = len(scores)
    return {
        "clips": n, "atoms": n_atoms, "variants": n_variants, "rejected": n_rej,
        "mean_score": round(float(np.mean(scores)), 1) if n else None,
        "min_score": round(float(np.min(scores)), 1) if n else None,
        "grades": {k: g.get(k, 0) for k in "ABCD"},
        "low_share": round((g.get("C", 0) + g.get("D", 0)) / n, 3) if n else None,
        "flags": dict(fl.most_common()),
    }


def build_tree(recordings: dict[str, dict], rejected: list[dict], atoms: list[dict], variants: list[dict],
               dataset_name: str = "sonic_for_a3 · selected20") -> dict:
    """atoms/variants: dicts with id, parent(source/atom), label, hand, duration, score, grade, flags, ops."""
    nodes: dict[str, dict] = {}

    def add(nid, parent, kind, name, **meta):
        nodes[nid] = {"id": nid, "parent": parent, "kind": kind, "name": name, "meta": meta, "children": []}
        if parent:
            nodes[parent]["children"].append(nid)

    add("lib", None, "library", "A3 手势动作库")
    add("ds:real", "lib", "dataset", dataset_name,
        note="A3 真机动捕重定向，120 Hz，38 列 CSV；本次只用了仓库公开的 20 条")
    for rec, st in recordings.items():
        add(f"rec:{rec}", "ds:real", "recording", rec, **st)
    for i, r in enumerate(rejected):
        src = r["source"]
        span = r.get("span_s")
        add(f"rej:{src}:{i}", f"rec:{src}", "rejected",
            f"丢弃 {span[0]:.2f}–{span[1]:.2f}s" if span else "整条丢弃", reason=r["reason"])
    for a in atoms:
        add(a["id"], f"rec:{a['source']}", "atom", a["id"].split("__")[-1] + " s",
            label=a["label"], hand=a["hand"], duration=a["duration"], score=a["score"], grade=a["grade"],
            flags=a["flags"], repairs=a.get("repairs", []))
    for v in variants:
        add(v["id"], v["parent"], "variant", v["style_zh"], style=v["style"], label=v["label"], hand=v["hand"],
            duration=v["duration"], score=v["score"], grade=v["grade"], flags=v["flags"],
            ops=v.get("ops", {}), repairs=v.get("repairs", []), accepted=v.get("accepted", True))

    # rollups bottom-up
    def walk(nid) -> tuple[list, list, list, int, int, int]:
        nd = nodes[nid]
        S, G, F, rej, na, nv = [], [], [], 0, 0, 0
        if nd["kind"] in ("atom", "variant") and nd["meta"].get("accepted", True):
            S.append(nd["meta"]["score"]); G.append(nd["meta"]["grade"]); F.append(nd["meta"]["flags"])
            na += nd["kind"] == "atom"; nv += nd["kind"] == "variant"
        if nd["kind"] == "rejected" or (nd["kind"] == "variant" and not nd["meta"].get("accepted", True)):
            rej += 1
        for c in nd["children"]:
            s, g, f, r, a_, v_ = walk(c)
            S += s; G += g; F += f; rej += r; na += a_; nv += v_
        nd["rollup"] = _rollup(S, G, F, rej, na, nv)
        return S, G, F, rej, na, nv

    walk("lib")

    # a second view: the same clips grouped by style (where does each style fail?)
    by_style: dict[str, dict] = {}
    for v in variants:
        k = v["style_zh"] if not v["style"].startswith("ref:") else "参考风格迁移"
        by_style.setdefault(k, {"S": [], "G": [], "F": [], "rej": 0})
        if v.get("accepted", True):
            by_style[k]["S"].append(v["score"]); by_style[k]["G"].append(v["grade"]); by_style[k]["F"].append(v["flags"])
        else:
            by_style[k]["rej"] += 1
    style_view = {k: _rollup(d["S"], d["G"], d["F"], d["rej"], 0, len(d["S"])) for k, d in by_style.items()}
    base = [a for a in atoms]
    style_view["原始（未风格化）"] = _rollup([a["score"] for a in base], [a["grade"] for a in base],
                                      [a["flags"] for a in base], 0, len(base), 0)
    return {"root": "lib", "nodes": nodes, "by_style": style_view}
