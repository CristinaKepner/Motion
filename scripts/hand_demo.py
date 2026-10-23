"""Dexterous-hand demo: library build + annotation + spec check, per-attribute stylisation,
streaming fusion agent (timestamps, response time, seek), and the three scripted scenes:
两手「1+1=2」 · 石头剪刀布 · 回放第五秒.

usage: python scripts/hand_demo.py [--no-video]
outputs: results/hand_library/{library.json, poses_sheet.png, style_curves.png}
         results/hand_demo/{timeline.json, demo.mp4, seek_5s.png, seek_5s.json, contact_sheet.png, stream_log.jsonl}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from trabot_a3.data import ALL_JOINTS, NEUTRAL_RAD  # noqa: E402
from trabot_a3.hand import (HAND_PRESETS, O10_CALIBRATION_NOTE, O10_CHANNELS, POSES, HandModel, make_atom)  # noqa: E402
from trabot_a3.hand_agent import FPS, Event, FusionAgent, script_count, script_one_plus_one, script_rps, save_timeline  # noqa: E402
from trabot_a3.hand_render import HandScene, check_hand_spec  # noqa: E402
from trabot_a3.paths import OUT  # noqa: E402

LIB = OUT / "hand_library"
DEMO = OUT / "hand_demo"
J = ALL_JOINTS.index
ARM = ["_shoulder_pitch_joint", "_shoulder_roll_joint", "_shoulder_yaw_joint", "_elbow_joint", "_wrist_roll_joint", "_wrist_pitch_joint", "_wrist_yaw_joint"]
# arm pose that presents both palms to the viewer (IK-solved once; see docs/HAND.md)
PRESENT = {"L": [-0.554, 0.211, -0.191, -0.231, 1.423, 0.271, 0.839], "R": [-0.552, -0.213, 0.187, -0.235, -1.422, 0.27, -0.838]}


def present_pose() -> np.ndarray:
    q = NEUTRAL_RAD.copy()
    for pre, side in (("left", "L"), ("right", "R")):
        q[[J(pre + s) for s in ARM]] = PRESENT[side]
    return q


def font(size=18):
    for p in ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ----------------------------------------------------------------------------- 1. library + annotation + 名实核对
def build_library(sc: HandScene):
    LIB.mkdir(parents=True, exist_ok=True)
    atoms, spec_fail = [], 0
    for label, p in POSES.items():
        for side in "LR":
            spec = check_hand_spec(sc, side, p["u"], p["spec"])
            for st in HAND_PRESETS.values():
                a = make_atom(label, side, st=st)
                # the stylised target is what gets checked (tension / spread can break a spec)
                sp2 = check_hand_spec(sc, side, a.keys[-2].u, p["spec"])
                spec_fail += not sp2["pass"]
                m = a.meta()
                m["annotation"].update({"spec_pass_base": spec["pass"], "spec_pass_styled": sp2["pass"],
                                        "spec_checks": sp2["checks"], "fingertips_palm_m": sp2["tips_palm_frame"],
                                        "quality": {"max_channel_rate_per_s": round(float(np.max(np.abs(np.diff(a.frames(), axis=0)) * FPS)), 3)}})
                atoms.append(m)
    lib = {"o10_channels": O10_CHANNELS, "o10_range": [0, 4096], "calibration_note": O10_CALIBRATION_NOTE,
           "poses": {k: {"zh": v["zh"], "u": np.round(v["u"], 3).tolist(), "o10": HandModel.to_o10(v["u"]), "spec": v["spec"]} for k, v in POSES.items()},
           "styles": {k: {"zh": v.zh, **v.ops()} for k, v in HAND_PRESETS.items()},
           "atoms": atoms, "n_atoms": len(atoms), "n_spec_fail_styled": spec_fail}
    (LIB / "library.json").write_text(json.dumps(lib, ensure_ascii=False, indent=1))
    print(f"hand library: {len(atoms)} atoms ({len(POSES)} poses × 2 sides × {len(HAND_PRESETS)} styles); styled spec failures: {spec_fail}")

    # pose sheet (right hand, presenting pose)
    sc.set_body(present_pose())
    tiles = []
    f = font(22)
    for name, p in POSES.items():
        sc.set_hand("R", p["u"]); sc.set_hand("L", p["u"]); sc.forward()
        pp, _ = sc.palm("R")
        im = Image.fromarray(sc.render(lookat=pp + [0, 0, 0.05], azimuth=180, elevation=-5, distance=0.4))
        ImageDraw.Draw(im).text((10, 8), f"{name}  {p['zh']}", fill=(255, 230, 80), font=f)
        tiles.append(np.array(im))
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 20, np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
    Image.fromarray(sheet).save(LIB / "poses_sheet.png")

    # style curves: index_bent over time for the same gesture under each preset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, st in HAND_PRESETS.items():
        a = make_atom("two", "R", st=st)
        fr = a.frames()
        t = np.arange(len(fr)) / FPS
        ax[0].plot(t, fr[:, 7], label=f"{k} {st.zh}")       # ring_bent (curls for "two")
        ax[1].plot(t, fr[:, 3], label=f"{k} {st.zh}")       # index_wiggles (spread)
    ax[0].set_title("ring_bent — tempo / snap / hold / anticipation / ripple"); ax[1].set_title("index_wiggles — spread")
    for a_ in ax:
        a_.set_xlabel("s"); a_.set_ylim(-0.05, 1.05); a_.grid(alpha=.3)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(LIB / "style_curves.png", dpi=130)
    return lib


# ----------------------------------------------------------------------------- 2. streaming scenes
def build_scenes() -> FusionAgent:
    ag = FusionAgent(body_library={"present": present_pose()[None, :]})
    ag.submit(Event(0.0, "body", None, "R", body_label="present", event_id="body_present", text="抬手"))  # body only
    script_one_plus_one(ag, t0=2.2, style="neutral", beat=0.9)                 # 2.2 … ~5.9 s (body is in "present" by ~1.5 s)
    script_rps(ag, t0=6.6, choice="scissors", style="lively", beat=0.6, side="R")  # 6.6 … reveal at 8.4
    script_rps(ag, t0=6.6, choice="paper", style="calm", beat=0.6, side="L")       # the other hand plays too
    script_count(ag, n=5, t0=10.6, beat=0.7, style="enthusiastic", side="R")       # 10.6 … 13.4 s
    # a late event: arrives 60 ms before its target → agent compresses tempo and logs late_ms
    ag.submit(Event(14.2, "hand", "ok", "L", t_target=14.26, style="formal", hold_s=0.8, text="OK（迟到事件）", event_id="late_ok"))
    return ag


def overlay(frame: np.ndarray, state: dict, ag: FusionAgent, f_big, f_small) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    t = state["t"]
    d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
    d.text((10, 6), f"t = {t:6.2f} s   A3 + OmniHand(O10)   流式 30 Hz", fill=(255, 255, 255), font=f_big)
    y = im.height - 150
    d.rectangle([0, y, im.width, im.height], fill=(0, 0, 0))
    for col, side in ((10, "L"), (im.width // 2 + 10, "R")):
        act = state["active"][side]
        ev = next((e for e in ag.log if e["event"]["event_id"] == act and e["event"]["side"] == side), None)
        txt = f"{side}: {ev['event']['text']} [{ev['event']['hand_label']}/{ev['event']['style']}]  resp {ev['response_ms']:.0f} ms" if ev else f"{side}: —"
        if ev and ev["late_ms"] > 0:
            txt += f"  late {ev['late_ms']:.0f} ms ×{ev['tempo_boost']:.2f}"
        d.text((col, y + 6), txt, fill=(255, 230, 80), font=f_small)
        o10 = state[f"{side}_o10"]
        for i, (ch, v) in enumerate(zip(O10_CHANNELS, o10)):
            bx = col + i * 44
            d.rectangle([bx, y + 118 - 80 * v / 4096, bx + 36, y + 118], fill=(90, 170, 255))
            d.text((bx, y + 122), ch.split("_")[0][:3] + ("B" if "bent" in ch else "W" if "wig" in ch else "R"), fill=(200, 200, 200), font=f_small)
    # event markers: upcoming targets
    for e in ag.log:
        tt = e["event"]["t_target"]
        if tt and 0 <= tt - t < 1.0:
            d.text((im.width - 240, 40 + 18 * int((tt - t) * 4)), f"→ {e['event']['text']} @ {tt:.2f}", fill=(150, 255, 150), font=f_small)
    return np.array(im)


def render_demo(ag: FusionAgent, sc: HandScene, video: bool):
    DEMO.mkdir(parents=True, exist_ok=True)
    save_timeline(ag, DEMO / "timeline.json")
    f_big, f_small = font(20), font(14)
    frames, stream_log = [], []
    t_end = ag.t_end
    wall0 = time.perf_counter()
    for st in ag.ticks(t_end):
        tick0 = time.perf_counter()
        sc.set_body(st["body"]); sc.set_hand("L", st["L"]); sc.set_hand("R", st["R"]); sc.forward()
        stream_log.append({"t": round(st["t"], 3), "L_o10": st["L_o10"], "R_o10": st["R_o10"], "active": st["active"],
                           "tick_ms": round((time.perf_counter() - tick0) * 1000, 3)})
        if video or abs(st["t"] - round(st["t"] * 2) / 2) < 1e-6:
            fr = sc.render(lookat=[0.2, 0, 1.25], azimuth=180, elevation=-8, distance=1.25)
            frames.append((st["t"], overlay(fr, st, ag, f_big, f_small)))
    (DEMO / "stream_log.jsonl").write_text("\n".join(json.dumps(x) for x in stream_log))
    print(f"streamed {len(stream_log)} ticks ({t_end:.1f} s) in {time.perf_counter()-wall0:.1f} s wall; "
          f"mean tick {np.mean([x['tick_ms'] for x in stream_log]):.2f} ms (FK only, no render)")
    if video:
        import imageio
        with imageio.get_writer(DEMO / "demo.mp4", fps=int(FPS), codec="libx264", quality=8, macro_block_size=1) as w:
            for _, fr in frames:
                w.append_data(fr)
        print("video", DEMO / "demo.mp4")
    # contact sheet: key moments
    keys = [2.2, 3.1, 4.0, 5.0, 7.2, 8.0, 8.45, 10.6, 12.0, 13.4, 14.4, 15.0]
    tiles = []
    for tk in keys:
        st = ag.at(tk)
        sc.set_body(st["body"]); sc.set_hand("L", st["L"]); sc.set_hand("R", st["R"]); sc.forward()
        tiles.append(overlay(sc.render(lookat=[0.2, 0, 1.25], azimuth=180, elevation=-8, distance=1.25), st, ag, f_big, f_small))
    h, w = tiles[0].shape[:2]
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t_ in enumerate(tiles):
        sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t_
    Image.fromarray(sheet).save(DEMO / "contact_sheet.png")
    # 回放第五秒: seek without stepping through the stream
    t0 = time.perf_counter()
    st = ag.at(5.0)
    seek_ms = (time.perf_counter() - t0) * 1000
    sc.set_body(st["body"]); sc.set_hand("L", st["L"]); sc.set_hand("R", st["R"]); sc.forward()
    Image.fromarray(overlay(sc.render(lookat=[0.2, 0, 1.25], azimuth=180, elevation=-8, distance=1.25), st, ag, f_big, f_small)).save(DEMO / "seek_5s.png")
    (DEMO / "seek_5s.json").write_text(json.dumps({"t": 5.0, "seek_ms": round(seek_ms, 3), "L_o10": st["L_o10"], "R_o10": st["R_o10"],
                                                   "active": st["active"], "body_deg": np.round(np.degrees(st["body"]), 2).tolist()}, ensure_ascii=False, indent=1))
    print(f"seek at(5.0): {seek_ms:.2f} ms, active {st['active']}, R_o10 {st['R_o10']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    sc = HandScene(960, 720)
    build_library(sc)
    ag = build_scenes()
    tl = ag.timeline()
    print("response:", tl["response_ms"], "| events:", len(tl["events"]))
    render_demo(ag, sc, not a.no_video)


if __name__ == "__main__":
    main()
