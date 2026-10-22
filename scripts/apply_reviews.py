"""Apply human reviews (from the review page) to produce the next catalog version.

Input: a reviews JSON exported from the page (or read from its shared store), a list of
  {clip_id, reviewer, naturalness 1-5, coherence 1-5, semantic 1-5, style_fit 1-5,
   decision: "accept"|"trim"|"reject", trim_in, trim_out (seconds, optional),
   label (optional corrected semantic function), note}

Per clip, reviews are aggregated (mean ratings, majority decision, median trim). Accepted /
trimmed clips are cut ("clip 出来"), re-validated with the same Eq.10 checks and re-scored,
and written to results/catalog_v2 with full provenance (source clip, reviewers, ratings, trim,
label correction). Rejected clips are listed with the reviewers' notes, so the next
generation round can be steered away from them.

usage: python scripts/apply_reviews.py reviews.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import numpy as np  # noqa: E402

from trabot_a3.atoms import load_catalog, kinematic_report, CatalogParams, SEMANTIC_FUNCTIONS  # noqa: E402
from trabot_a3.paths import MJCF, DATA, OUT as OUT_ROOT  # noqa: E402
from trabot_a3.quality import PhysicsProbe, evaluate_clip  # noqa: E402


def load_q(clip_id: str, cat) -> tuple[np.ndarray, float]:
    if "~" in clip_id:
        z = np.load(OUT_ROOT / "library" / f"{clip_id.replace('~', '__STYLE__')}.npz")
        return z["q"], 30.0
    a = next(a for a in cat.atoms if a.atom_id == clip_id)
    return a.q, a.fps


def main(path: str):
    reviews = json.loads(Path(path).read_text())
    bundle = json.loads((OUT_ROOT / "library" / "review_bundle.json").read_text())
    info = {c["id"]: c for c in bundle["clips"]}
    cat = load_catalog(OUT_ROOT / "catalog")
    probe = PhysicsProbe(MJCF)
    p = CatalogParams()
    by_clip: dict[str, list[dict]] = {}
    for r in reviews:
        by_clip.setdefault(r["clip_id"], []).append(r)

    out_dir = OUT_ROOT / "catalog_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    for cid, rs in by_clip.items():
        if cid not in info:
            continue
        dec = Counter(r.get("decision") for r in rs).most_common(1)[0][0]
        ratings = {k: round(float(np.mean([r[k] for r in rs if r.get(k)])), 2)
                   for k in ("naturalness", "coherence", "semantic", "style_fit") if any(r.get(k) for r in rs)}
        labels = [r["label"] for r in rs if r.get("label") in SEMANTIC_FUNCTIONS]
        label = Counter(labels).most_common(1)[0][0] if labels else info[cid]["label"]
        notes = [f'{r.get("reviewer","?")}: {r["note"]}' for r in rs if r.get("note")]
        if dec == "reject" or dec is None:
            rejected.append({"clip_id": cid, "ratings": ratings, "notes": notes, "flags": info[cid]["flags"]})
            continue
        q, fs = load_q(cid, cat)
        t_in = float(np.median([r["trim_in"] for r in rs if r.get("trim_in") is not None] or [0.0]))
        t_out = float(np.median([r["trim_out"] for r in rs if r.get("trim_out") is not None] or [q.shape[0] / fs]))
        a, b = max(0, int(round(t_in * fs))), min(q.shape[0], int(round(t_out * fs)))
        if b - a < int(p.min_atom_s * fs):
            rejected.append({"clip_id": cid, "ratings": ratings, "notes": notes + ["裁剪后短于 0.4 s"]})
            continue
        qc = q[a:b]
        rep = kinematic_report(qc, fs, p)
        if rep.limit_violation > 0 or rep.stretch > 1.0 + 1e-3:
            rejected.append({"clip_id": cid, "ratings": ratings, "notes": notes + ["裁剪后未过运动学校验"]})
            continue
        qual, _ = evaluate_clip(cid, qc, fs, probe)
        new_id = f"{cid}@{t_in:.2f}-{t_out:.2f}" if dec == "trim" else cid
        np.savez_compressed(out_dir / f"{new_id.replace('~', '__STYLE__')}.npz", q=qc)
        accepted.append({
            "atom_id": new_id, "label": label, "label_changed": label != info[cid]["label"],
            "hand": info[cid]["hand"], "duration": round(qc.shape[0] / fs, 3), "fps": fs,
            "source_clip": cid, "kind": info[cid]["kind"], "style": info[cid]["style"],
            "trim_s": [round(t_in, 3), round(t_out, 3)] if dec == "trim" else None,
            "reviewed": True, "reviewers": sorted({r.get("reviewer", "?") for r in rs}), "ratings": ratings,
            "auto_score": qual.score, "auto_grade": qual.grade, "auto_flags": qual.flags, "notes": notes})
    # spec calibration: how often reviewers think the automatic semantic check was right,
    # per function — the data for tuning trabot_a3/semantics.py thresholds
    calib: dict[str, dict[str, int]] = {}
    for r in reviews:
        if r.get("spec_agree") and r.get("clip_id") in info:
            fn = info[r["clip_id"]]["label"]
            calib.setdefault(fn, {"agree": 0, "false_pass": 0, "false_fail": 0})[r["spec_agree"]] += 1
    (out_dir / "catalog.json").write_text(json.dumps({"atoms": accepted, "rejected_by_review": rejected,
                                                     "spec_calibration": calib},
                                                     indent=1, ensure_ascii=False))
    print(f"reviewed clips {len(by_clip)}: accepted {len(accepted)} "
          f"(trimmed {sum(1 for a in accepted if a['trim_s'])}, relabelled {sum(1 for a in accepted if a['label_changed'])}), "
          f"rejected {len(rejected)} → {out_dir}")
    if calib:
        print("spec calibration (agree / false_pass / false_fail):",
              {k: (v["agree"], v["false_pass"], v["false_fail"]) for k, v in calib.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(OUT_ROOT / "reviews.json"))
