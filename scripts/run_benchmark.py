"""Run the gesture benchmark.

  python scripts/run_benchmark.py library                      # A3 library: atoms vs stylised variants (+ metric self-checks)
  python scripts/run_benchmark.py bvh <dir> --profile beat     # a public dataset folder (BVH + optional WAV/TSV) → A3 → metrics
  python scripts/run_benchmark.py compare <dirA> <dirB> --profile beat  # e.g. generator output vs held-out data (FKD etc.)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from trabot_a3.paths import MJCF, OUT  # noqa: E402
from trabot_a3.quality import PhysicsProbe  # noqa: E402
from benchmark import runner as RN, metrics as M  # noqa: E402

BOUT = OUT / "benchmark"


def self_checks(atoms: list[RN.Clip]) -> dict:
    """Metric sanity checks that need no audio: BC/BAS with beats placed on / off the motion."""
    rng = np.random.default_rng(0)
    on, off, b_on, b_off = [], [], [], []
    for c in atoms[:40]:
        mb = M.motion_beats(c.q); ap = M.hand_apexes(c.kp)
        if len(mb) < 2 or len(ap) < 1:
            continue
        T = c.q.shape[0] / 30.0
        on.append(M.beat_consistency(mb, mb)); off.append(M.beat_consistency(rng.uniform(0, T, len(mb)), mb))
        b_on.append(M.bas(ap, ap)[0]); b_off.append(M.bas(rng.uniform(0, T, len(ap)), ap)[0])
    # FKD: same distribution (two random halves) vs. a different one
    qs = [c.q for c in atoms]
    idx = list(range(len(qs))); random.Random(0).shuffle(idx)
    h1, h2 = [qs[i] for i in idx[::2]], [qs[i] for i in idx[1::2]]
    return {"BC_beats_on_motion": round(float(np.mean(on)), 3), "BC_random_beats": round(float(np.mean(off)), 3),
            "BAS_binary_on_apex": round(float(np.mean(b_on)), 3), "BAS_binary_random": round(float(np.mean(b_off)), 3),
            "FKD_same_distribution_halves": round(M.fkd(h1, h2), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["library", "bvh", "compare"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--profile", default="beat")
    ap.add_argument("--clip-seconds", type=float, default=8.0)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--no-physics", action="store_true")
    args = ap.parse_args()
    BOUT.mkdir(parents=True, exist_ok=True)
    probe = PhysicsProbe(MJCF)

    if args.mode == "library":
        atoms, vars_ = RN.clips_from_library("atom"), RN.clips_from_library("variant")
        ra = RN.evaluate(atoms, probe, name="A3 原始 atom（20 条智元录制）")
        rv = RN.evaluate(vars_, probe, name="A3 仿真风格变体")
        out = {"sources": [ra, rv], "compare": {"FKD(variants → atoms)": M.fkd([c.q for c in vars_], [c.q for c in atoms])},
               "self_checks": self_checks(atoms)}
        out["compare"] = {k: round(v, 3) for k, v in out["compare"].items()}
        (BOUT / "library_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "sources"}, ensure_ascii=False, indent=1))
        for s in out["sources"]:
            print(s["name"], s["executability"], "div", s["diversity_l1"], "sem", s["semantics"]["functions_with_refs"], "/17")
    elif args.mode == "bvh":
        clips, info = RN.clips_from_bvh_dir(args.paths[0], args.profile, args.clip_seconds, args.max_files)
        rep = RN.evaluate(clips, probe, physics=not args.no_physics, name=f"{args.profile}:{args.paths[0]}")
        rep["adaptation"] = info
        (BOUT / f"report_{args.profile}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    else:
        A, ia = RN.clips_from_bvh_dir(args.paths[0], args.profile, args.clip_seconds, args.max_files)
        B, ib = RN.clips_from_bvh_dir(args.paths[1], args.profile, args.clip_seconds, args.max_files)
        ra, rb = RN.evaluate(A, probe, physics=not args.no_physics, name=args.paths[0]), RN.evaluate(B, probe, physics=not args.no_physics, name=args.paths[1])
        out = {"A": ra, "B": rb, "compare": RN.compare(A, B), "adaptation": {"A": ia, "B": ib}}
        (BOUT / "compare_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(json.dumps(out["compare"], indent=1))


if __name__ == "__main__":
    main()
