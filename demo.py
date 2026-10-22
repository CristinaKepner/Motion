"""End-to-end demo: A3 CSV → Motion Atom catalog → budgeted semantic planning →
state-aware execution in MuJoCo (joint-angle playback) → mp4 + CSV + trace.

usage: python demo.py [--no-video]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from trabot_a3.atoms import build_catalog, load_catalog
from trabot_a3.data import A3Motion, NEUTRAL_ROOT, write_a3_csv
from trabot_a3.executor import StateAwareExecutor
from trabot_a3.planner import Planner, PlannerParams
from trabot_a3.sim import MujocoKinematicBackend

ROOT = Path(__file__).resolve().parent
from trabot_a3.paths import MJCF, DATA, OUT as OUT_ROOT
OUT = OUT_ROOT


def overlay(frames: list[np.ndarray], records, events: list[dict], fps: float) -> list[np.ndarray]:
    """Burn the current phase / atom / label into each frame (needs Pillow)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frames
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except Exception:
        font = font_s = ImageFont.load_default()
    atom_label = {e["atom"]: f'{e["label"]} ({e["hand"]})' for e in events if e["event"] == "atom"}
    out = []
    for fr, rec in zip(frames, records):
        im = Image.fromarray(fr)
        dr = ImageDraw.Draw(im)
        tag = rec.tag
        if tag.startswith("bridge"):
            phase, detail = "BRIDGE (quintic, Eq.13-14)", tag.split("→", 1)[1]
        elif tag == "neutral_recovery":
            phase, detail = "NEUTRAL RECOVERY (C_ret)", ""
        else:
            phase, detail = f"ATOM  {atom_label.get(tag, '')}", tag
        dr.rectangle([0, 0, im.width, 54], fill=(14, 17, 22))
        dr.text((12, 8), f"t={rec.t:5.2f}s  {phase}", fill=(76, 201, 240), font=font)
        dr.text((12, 32), detail, fill=(195, 204, 214), font=font_s)
        dr.text((im.width - 330, 32), "A3 · TRABot-style gesture planner · kinematic playback",
                fill=(139, 152, 168), font=font_s)
        out.append(np.asarray(im))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tau", type=float, default=0.12, help="first-order tracking lag (s) emulating the controller")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # ---------------- offline: catalog
    cat_dir = OUT / "catalog"
    if args.rebuild or not (cat_dir / "catalog.json").exists():
        t0 = time.time()
        cat = build_catalog(sorted(glob.glob(str(DATA / "*.csv"))), verbose=False)
        cat.save(cat_dir)
        print(f"catalog built in {time.time()-t0:.1f}s")
    else:
        cat = load_catalog(cat_dir)
    print(f"catalog: {len(cat.atoms)} atoms, {len(cat.rejected)} rejected; labels={cat.labels()}")

    # ---------------- online: planner + executor on MuJoCo
    planner = Planner(cat, PlannerParams(beam_width=32))
    backend = MujocoKinematicBackend(MJCF, fps=30.0, tracking_tau=args.tau, render=not args.no_video)
    ex = StateAwareExecutor(planner, backend)

    # Reply 1 — the paper's teaching example: "introduce AI" → turn_open → emphasize → depict_size_shape,
    # T_hat = 1.5 s so B = 4.5 s. Then the same request with a longer T_hat to show the prefix growing.
    cases = [
        ("reply-18", ["turn_open", "emphasize", "depict_size_shape"], 1.5),
        ("reply-19", ["turn_open", "emphasize", "depict_size_shape"], 3.0),
        ("reply-20", ["enumerate", "enumerate", "enumerate", "uncertainty"], 2.5),
    ]
    traces = []
    for reply_id, fns, T_hat in cases:
        tr = ex.run(reply_id, fns, T_hat)
        traces.append(tr)
        print("\n" + tr.summary())

    # Reply 21 — cancel mid-sequence (new turn arrives): epoch bump, stop after current atom, recover.
    import threading
    reply_id, fns, T_hat = "reply-21", ["emphasize", "refer_external", "depict_direction_motion"], 4.0
    orig_play = backend.play
    counter = {"atoms": 0}

    def play_with_interrupt(q_full, tag):
        r = orig_play(q_full, tag)
        if not tag.startswith("bridge") and tag != "neutral_recovery":
            counter["atoms"] += 1
            if counter["atoms"] == 1:
                ex.cancel()           # user barges in right after the first atom
        return r
    backend.play = play_with_interrupt
    tr = ex.run(reply_id, fns, T_hat)
    backend.play = orig_play
    traces.append(tr)
    print("\n" + tr.summary())

    # ---------------- outputs
    cmd = backend.commanded_trajectory()
    meas = backend.measured_trajectory()
    root = np.repeat(NEUTRAL_ROOT[None, :], cmd.shape[0], axis=0)
    write_a3_csv(OUT / "planned_sequence_30hz.csv", A3Motion("planned", 30.0, root, cmd))
    write_a3_csv(OUT / "measured_sequence_30hz.csv", A3Motion("measured", 30.0, root, meas))
    (OUT / "trace.json").write_text(json.dumps(
        [{"reply_id": t.reply_id, "epoch": t.epoch, "T_hat": t.T_hat, "budget": t.budget,
          "functions_requested": t.functions_requested, "functions_completed": t.functions_completed,
          "stop_reason": t.stop_reason, "elapsed_total": round(t.elapsed_total, 3), "events": t.events}
         for t in traces], indent=1, ensure_ascii=False))
    print(f"\ntrajectory frames: {cmd.shape[0]} ({cmd.shape[0]/30:.1f}s)  max |cmd-meas| arm = "
          f"{np.abs(cmd-meas)[:, 5:19].max():.3f} rad")

    if not args.no_video:
        import imageio
        frames = overlay(backend.frames, backend.records, [e for t in traces for e in t.events], 30.0)
        imageio.mimwrite(OUT / "demo.mp4", frames, fps=30, codec="libx264", quality=8, macro_block_size=1)
        print("video:", OUT / "demo.mp4")


if __name__ == "__main__":
    main()
