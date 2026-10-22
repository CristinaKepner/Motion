"""Build the evaluated, stylised motion library and the review payload.

  catalog (recorded atoms)
    → quality evaluation (naturalness / coherence / executability / expressiveness)
    → pick generation bases (good, non-static, diverse, round-robin over labels)
    → sim stylisation: 5 presets + mirror + reference-style transfer
    → same validation/repair + same evaluation for every variant
    → library richness report (atoms only vs. atoms + variants)
    → sourcing tree + review payload (stick-figure keypoints @ 15 fps)

usage: python build_library.py [--max-bases 36]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trabot_a3.atoms import load_catalog, SEMANTIC_FUNCTIONS
from trabot_a3.quality import PhysicsProbe, evaluate_clip, library_report, embed, FLAG_TEXT, style_descriptor
from trabot_a3.sourcing import build_tree
from trabot_a3.stylize import PRESETS, MIRROR, generate_variant, style_from_reference
from trabot_a3.semantics import check_all, SPEC_TEXT, FN_ZH

ROOT = Path(__file__).resolve().parent
from trabot_a3.paths import MJCF, DATA, OUT as OUT_ROOT
OUT = OUT_ROOT / "library"
KEEP_METRICS = ["duration", "sparc", "hf_rms", "hf_ratio", "jerk_ratio", "step_max", "v_entry", "v_exit",
                "bridge_in_s", "bridge_out_s", "limit_margin", "collision_excess_mm", "torque_ratio",
                "torque_joint", "track_rmse", "sat_frac", "extent", "peak_hand_speed", "periodicity"]


def kp_payload(kp: np.ndarray, stride: int = 2) -> list[int]:
    """(T,15,3) m → flat int mm list at 30/stride fps, centred on the first pelvis position."""
    k = kp[::stride] - kp[0:1, 0:1, :] * np.array([1, 1, 0])
    return np.round(k * 1000).astype(int).ravel().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bases", type=int, default=36)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cat = load_catalog(OUT_ROOT / "catalog")
    meta = json.loads((OUT_ROOT / "catalog" / "catalog.json").read_text())
    probe = PhysicsProbe(MJCF)

    # ---------------------------------------------------------------- 1. evaluate recorded atoms
    clips: list[dict] = []
    for a in cat.atoms:
        q, kp = evaluate_clip(a.atom_id, a.q, a.fps, probe)
        repairs = [n for n in a.notes if not n.startswith("label proposal")]
        clips.append(dict(id=a.atom_id, kind="atom", parent=None, source=a.source, label=a.label, hand=a.hand,
                          style="original", style_zh="原始", duration=round(a.duration, 3), fs=a.fps,
                          quality=q, kp=kp, q=a.q, repairs=repairs, ops={}, label_note=a.notes[0]))
    print(f"evaluated {len(clips)} atoms in {time.time()-t0:.0f}s")

    # ---------------------------------------------------------------- 2. choose generation bases
    good = [c for c in clips if c["quality"].grade in "AB"
            and not set(c["quality"].flags) & {"static", "spike", "self_collision", "torque"}]
    E = {c["id"]: embed(c["kp"]) for c in good}
    by_label: dict[str, list[dict]] = {}
    for c in sorted(good, key=lambda c: -c["quality"].score):
        by_label.setdefault(c["label"], []).append(c)
    bases: list[dict] = []
    while len(bases) < args.max_bases and any(by_label.values()):
        for lab in list(by_label):
            if not by_label[lab] or len(bases) >= args.max_bases:
                continue
            c = by_label[lab].pop(0)
            # skip near-duplicates of an already chosen base
            if any(np.sqrt(((E[c["id"]] - E[b["id"]]) ** 2).mean()) < 0.025 for b in bases):
                continue
            bases.append(c)
    print(f"bases: {len(bases)} from {len(good)} clean atoms; labels {sorted({b['label'] for b in bases})}")

    # references for style transfer: three recorded clips with clearly different character
    def pick(prefix):
        c = [x for x in clips if x["source"].startswith(prefix) and x["quality"].grade in "AB"]
        return max(c, key=lambda x: x["quality"].score) if c else None
    refs = [r for r in (pick("058_punch"), pick("077_reach_overhead"), pick("063_twist")) if r]
    atom_by_id = {a.atom_id: a for a in cat.atoms}

    # ---------------------------------------------------------------- 3. generate variants
    variants: list[dict] = []
    rejected_variants: list[dict] = []
    for bi, b in enumerate(bases):
        a = atom_by_id[b["id"]]
        styles = list(PRESETS.values())
        lp, rp = b["quality"].metrics["hand_path_L"], b["quality"].metrics["hand_path_R"]
        if abs(lp - rp) / max(lp, rp, 1e-6) > 0.25 or b["hand"] != "B":
            styles.append(MIRROR)
        for r in refs:
            if r["id"] != b["id"] and bi % 3 == refs.index(r):
                styles.append(style_from_reference(a.q, atom_by_id[r["id"]].q, a.fps, r["source"].split("_", 1)[1]))
        for st in styles:
            v = generate_variant(a, st)
            if v is None:
                rejected_variants.append(dict(id=f"{a.atom_id}~{st.name}", parent=a.atom_id, style=st.name,
                                              style_zh=st.zh, reason="修复后仍超限 / 放慢倍率超界"))
                continue
            q, kp = evaluate_clip(v.variant_id, v.q, v.fps, probe,
                                  extra_flags=["style_eroded"] if v.style_eroded else None)
            # executability feedback: the dynamics check (fixed-root inverse dynamics against the
            # A3 torque limits) constrains the generator — over-limit variants are slowed down
            # (inertial torque ∝ 1/s²) and re-evaluated, and the repair is recorded.
            for _ in range(2):
                tr = q.metrics.get("torque_ratio", 0.0)
                if tr <= 1.0:
                    break
                s = 1.05 * float(np.sqrt(tr))
                T = v.q.shape[0]
                n_new = int(round(T * s))
                t_src, t_dst = np.linspace(0, 1, T), np.linspace(0, 1, n_new)
                v.q = np.stack([np.interp(t_dst, t_src, v.q[:, k]) for k in range(v.q.shape[1])], 1)
                v.repairs.append(f"力矩超限 ×{tr:.2f} → 放慢 ×{s:.2f}")
                v.style_eroded = v.style_eroded or st.tempo > 1.0
                q, kp = evaluate_clip(v.variant_id, v.q, v.fps, probe,
                                      extra_flags=["style_eroded"] if v.style_eroded else None)
            variants.append(dict(id=v.variant_id, kind="variant", parent=a.atom_id, source=a.source, label=v.label,
                                 hand=v.hand, style=st.name, style_zh=st.zh, duration=round(v.duration, 3), fs=v.fps,
                                 quality=q, kp=kp, q=v.q, repairs=v.repairs, ops=st.ops(), label_note=""))
            np.savez_compressed(OUT / f"{v.variant_id.replace('~', '__STYLE__')}.npz", q=v.q)
    print(f"variants: {len(variants)} accepted, {len(rejected_variants)} rejected  ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- 3b. semantic self-check
    # "动作名对了，动作却没做对": verify each clip against the checkable spec of its label,
    # and record which specs it actually satisfies (retrieval of verified references).
    for c in clips + variants:
        R = check_all(c["kp"], c["q"], c["fs"])
        own = R[c["label"]]
        c["spec"] = own.as_dict()
        c["spec_pass_fns"] = [fn for fn, r in R.items() if r.passed]
        c["spec_near_fns"] = sorted([(fn, r.score) for fn, r in R.items() if not r.passed and r.score >= 0.5],
                                    key=lambda x: -x[1])[:3]
        if not own.passed:
            c["quality"].flags.append("semantic_mismatch")
    sem = {}
    for fn in SPEC_TEXT:
        lab = [c for c in clips if c["label"] == fn]
        refs = sorted([c for c in clips + variants if fn in c["spec_pass_fns"] and c["quality"].grade in "AB"],
                      key=lambda c: -c["quality"].score)
        sem[fn] = {"zh": FN_ZH[fn], "spec": SPEC_TEXT[fn], "labelled": len(lab),
                   "labelled_pass": sum(1 for c in lab if fn in c["spec_pass_fns"]),
                   "verified_refs": len(refs), "verified_refs_atoms": sum(1 for c in refs if c["kind"] == "atom"),
                   "best_refs": [c["id"] for c in refs[:5]]}
    print("semantic self-check: own-label pass",
          sum(1 for c in clips if c["label"] in c["spec_pass_fns"]), "/", len(clips), "atoms;",
          sum(1 for c in variants if c["label"] in c["spec_pass_fns"]), "/", len(variants), "variants")

    # ---------------------------------------------------------------- 4. richness report
    rep_atoms = library_report(clips)
    usable_variants = [v for v in variants if v["quality"].grade in "AB"]
    rep_all = library_report(clips + usable_variants)
    report = {"atoms_only": rep_atoms, "atoms_plus_AB_variants": rep_all,
              "counts": {"atoms": len(clips), "variants": len(variants), "variants_AB": len(usable_variants),
                         "variants_rejected": len(rejected_variants), "bases": len(bases),
                         "recordings": len(meta["recordings"]), "segments_rejected": len(meta["rejected"])},
              "semantic_functions": SEMANTIC_FUNCTIONS, "semantic_check": sem}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # ---------------------------------------------------------------- 5. sourcing tree
    def flat(c):
        return dict(id=c["id"], parent=c["parent"], source=c["source"], label=c["label"], hand=c["hand"],
                    duration=c["duration"], score=c["quality"].score, grade=c["quality"].grade,
                    flags=c["quality"].flags, repairs=c["repairs"], ops=c["ops"], style=c["style"],
                    style_zh=c["style_zh"])
    tree = build_tree(meta["recordings"], meta["rejected"], [flat(c) for c in clips],
                      [flat(v) for v in variants] +
                      [dict(id=r["id"], parent=r["parent"], label="", hand="", duration=0, score=0, grade="D",
                            flags=[], style=r["style"], style_zh=r["style_zh"], accepted=False,
                            repairs=[r["reason"]]) for r in rejected_variants])
    (OUT / "tree.json").write_text(json.dumps(tree, ensure_ascii=False))

    # ---------------------------------------------------------------- 6. review payload
    payload = []
    for c in clips + variants:
        q = c["quality"]
        payload.append(dict(
            id=c["id"], kind=c["kind"], parent=c["parent"], source=c["source"], label=c["label"], hand=c["hand"],
            style=c["style"], style_zh=c["style_zh"], duration=c["duration"], fps=15,
            score=q.score, grade=q.grade, sub=q.sub, flags=q.flags,
            metrics={k: q.metrics.get(k) for k in KEEP_METRICS}, repairs=c["repairs"], ops=c["ops"],
            style_xy=c.get("style_xy") or [round(x, 3) for x in style_descriptor(c["kp"], c["fs"])],
            label_note=c["label_note"], spec=c["spec"], spec_pass=c["spec_pass_fns"], spec_near=c["spec_near_fns"],
            kp=kp_payload(c["kp"])))
    bundle = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "flag_text": FLAG_TEXT,
              "keypoint_names": ["pelvis", "torso", "head", "L_shoulder", "L_elbow", "L_hand", "R_shoulder",
                                 "R_elbow", "R_hand", "L_hip", "L_knee", "L_ankle", "R_hip", "R_knee", "R_ankle"],
              "report": report, "tree": tree, "clips": payload,
              "styles": {k: {"zh": s.zh, **s.ops()} for k, s in PRESETS.items()} | {"mirror": {"zh": "左右镜像"}},
              "spec_text": SPEC_TEXT}
    (OUT / "review_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    sz = (OUT / "review_bundle.json").stat().st_size / 1e6
    print(f"bundle: {len(payload)} clips, {sz:.1f} MB  ({time.time()-t0:.0f}s total)")

    ra, rb = rep_atoms, rep_all
    print("\n== richness: atoms only → atoms + A/B variants ==")
    print(f"clips {len(clips)} → {len(clips)+len(usable_variants)}")
    print(f"labels covered {ra['labels_covered']}/17 → {rb['labels_covered']}/17   balance {ra['label_balance']} → {rb['label_balance']}")
    print(f"hand {ra['hand']} → {rb['hand']}")
    print(f"style cells {ra['style_cells_covered']}/{ra['style_cells_total']} → {rb['style_cells_covered']}/{rb['style_cells_total']}")
    print(f"grades atoms {ra['grades']}   variants {library_report(variants)['grades'] if variants else {}}")
    print(f"top flags atoms {list(ra['flags'].items())[:6]}")
    print(f"near-duplicate atom pairs {ra['near_duplicate_count']}")
    print(f"intra-label diversity (m) atoms {ra['intra_label_diversity']}")
    print(f"reachable from neutral ≤0.6s {ra['reachable_from_neutral_0p6s']}  median bridge-in {ra['median_bridge_in_s']}s")


if __name__ == "__main__":
    main()
