"""Benchmark runner.

A *source* is one of
  * a folder of BVH files (+ optional .wav / .tsv with the same stem) with a skeleton profile → retargeted to A3
  * the A3 library itself (catalog atoms, stylised variants) → already in A3 joint space
Each source becomes a list of clips (A3 joint angles at 30 Hz, keypoints, optional beats, label), and the
same metric set is computed. Two sources can be compared with FKD (e.g. generator output vs. held-out data).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trabot_a3.atoms import load_catalog, build_catalog, CatalogParams
from trabot_a3.data import A3Motion, NEUTRAL_RAD, CTRL_IDX
from trabot_a3.paths import MJCF, OUT
from trabot_a3.quality import PhysicsProbe, evaluate_clip
from .bvh import load_bvh
from .retarget import retarget_bvh
from . import metrics as M


@dataclass
class Clip:
    id: str
    q: np.ndarray                  # (T,31) rad @30 Hz
    kp: np.ndarray | None = None   # (T,15,3)
    label: str | None = None
    audio_beats: np.ndarray | None = None
    stress: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def _resample30(q: np.ndarray, fps: float) -> np.ndarray:
    m = A3Motion("x", fps, np.zeros((q.shape[0], 6)), q).resample(30.0)
    return m.q


def clips_from_bvh_dir(folder: str, profile: str, clip_s: float = 8.0, max_files: int | None = None,
                       up_axis=np.array([0, 1.0, 0])) -> tuple[list[Clip], dict]:
    files = sorted(Path(folder).glob("*.bvh"))[:max_files]
    labels = {}
    lp = Path(folder) / "labels.json"
    if lp.exists():
        labels = {k: v.get("label") if isinstance(v, dict) else v for k, v in json.loads(lp.read_text()).items()}
    clips, fid = [], []
    t0 = time.time()
    for f in files:
        b = load_bvh(str(f))
        res = retarget_bvh(b, profile, up_axis=up_axis)
        fid.append(res.fidelity)
        q = _resample30(res.q, b.fps)
        beats = stress = None
        wav, tsv = f.with_suffix(".wav"), f.with_suffix(".tsv")
        if wav.exists():
            on, st = M.audio_onsets(str(wav)); beats, stress = on, st
        elif tsv.exists():
            beats = M.transcript_onsets(str(tsv)); stress = beats
        n = int(clip_s * 30)
        starts = [0] if q.shape[0] <= n else range(0, q.shape[0] - n + 1, n)
        for k in starts:
            seg = q[k:k + n]
            if seg.shape[0] < 30:
                continue
            t_off = k / 30.0
            cb = None if beats is None else beats[(beats >= t_off) & (beats < t_off + clip_s)] - t_off
            cs = None if stress is None else stress[(stress >= t_off) & (stress < t_off + clip_s)] - t_off
            clips.append(Clip(f"{f.stem}@{t_off:.1f}", seg, label=labels.get(f.name), audio_beats=cb, stress=cs,
                              meta={"file": f.name}))
    info = {"files": len(files), "retarget_fidelity": {k: round(float(np.mean([x[k] for x in fid])), 2) for k in fid[0]} if fid else None,
            "seconds": round(time.time() - t0, 1)}
    return clips, info


def clips_from_library(kind: str = "atom") -> list[Clip]:
    lib = OUT / "library"
    bundle = json.loads((lib / "review_bundle.json").read_text())
    cat = load_catalog(OUT / "catalog")
    aq = {a.atom_id: a.q for a in cat.atoms}
    out = []
    for c in bundle["clips"]:
        if c["kind"] != kind:
            continue
        q = aq[c["id"]] if kind == "atom" else np.load(lib / f"{c['id'].replace('~', '__STYLE__')}.npz")["q"]
        qe = np.repeat(NEUTRAL_RAD[None], q.shape[0], 0); qe[:, CTRL_IDX] = q[:, CTRL_IDX]
        out.append(Clip(c["id"], qe, label=c["label"], meta={"grade": c["grade"], "style": c["style"]}))
    return out


def evaluate(clips: list[Clip], probe: PhysicsProbe, physics: bool = True, name: str = "") -> dict:
    t0 = time.time()
    grades, flags = {"A": 0, "B": 0, "C": 0, "D": 0}, {}
    for c in clips:
        qual, kp = evaluate_clip(c.id, c.q, 30.0, probe, physics=physics)
        c.kp = kp
        c.meta["quality"] = qual
        grades[qual.grade] += 1
        for f in qual.flags:
            flags[f] = flags.get(f, 0) + 1
    n = max(len(clips), 1)
    kps, qs = [c.kp for c in clips], [c.q for c in clips]
    bc, bbin, bgau, bc0, bbin0 = [], [], [], [], []
    rng = np.random.default_rng(0)
    for c in clips:
        if c.audio_beats is not None and len(c.audio_beats):
            mb, ap = M.motion_beats(c.q), M.hand_apexes(c.kp)
            T = c.q.shape[0] / 30.0
            bc.append(M.beat_consistency(c.audio_beats, mb))
            st = c.stress if c.stress is not None else c.audio_beats
            b1, b2 = M.bas(st, ap)
            bbin.append(b1); bgau.append(b2)
            # chance level: the same number of beats at random times (BC saturates on dense motion beats)
            bc0.append(M.beat_consistency(rng.uniform(0, T, len(c.audio_beats)), mb))
            bbin0.append(M.bas(rng.uniform(0, T, len(st)), ap)[0])
    rep = {
        "name": name, "clips": len(clips), "seconds_of_motion": round(sum(q.shape[0] for q in qs) / 30.0, 1),
        "executability": {"grades": grades, "torque_over_limit_share": round(flags.get("torque", 0) / n, 3),
                          "self_collision_share": round(flags.get("self_collision", 0) / n, 3),
                          "static_share": round(flags.get("static", 0) / n, 3),
                          "jitter_share": round(flags.get("jitter", 0) / n, 3),
                          "mean_score": round(float(np.mean([c.meta["quality"].score for c in clips])), 1)},
        "diversity_l1": round(M.l1_diversity(kps), 4),
        "semantics": M.semantic_summary(kps, qs, [c.label for c in clips]),
        "joint_coverage": M.joint_coverage(qs),
        "style": M.style_cells(kps),
        "rhythm": {"clips_with_audio": len(bc),
                   "BC": round(float(np.nanmean(bc)), 3) if bc else None,
                   "BAS_binary": round(float(np.nanmean(bbin)), 3) if bbin else None,
                   "BAS_gaussian": round(float(np.nanmean(bgau)), 3) if bgau else None,
                   "BC_chance": round(float(np.nanmean(bc0)), 3) if bc0 else None,
                   "BAS_binary_chance": round(float(np.nanmean(bbin0)), 3) if bbin0 else None},
        "eval_seconds": round(time.time() - t0, 1),
    }
    return rep


def compare(a: list[Clip], b: list[Clip]) -> dict:
    return {"FKD": round(M.fkd([c.q for c in a], [c.q for c in b]), 3)}
