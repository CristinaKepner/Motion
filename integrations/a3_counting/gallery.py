"""Render actual MuJoCo hand tracking at the audience's three viewing angles.

Run: MUJOCO_GL=egl python -m a3_counting.gallery
Interactive tuning: python -m a3_counting.gallery --viewer (keys 0/1/2/3).
"""

import argparse
from .o10 import metadata as hand_metadata
import json
from pathlib import Path
import queue
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .hand import HandController, build_model, preset


def camera(data, model, azimuth=0, close=True):
    cam = mujoco.MjvCamera()
    cam.azimuth = 180 + azimuth
    cam.elevation = -5 if close else -8
    if close:
        cam.lookat[:] = data.site_xpos[model.site("right_palm_center").id] + [0, 0, .035]
        cam.distance = .53
    else:
        cam.lookat[:] = [.02, -.05, .96]
        cam.distance = 2.8
    return cam


def render(renderer, model, data, angle, close=True):
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    renderer.update_scene(data, camera=camera(data, model, angle, close), scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    return Image.fromarray(renderer.render())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/counting/o10/gallery"))
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    model, tree, arm_pose = build_model(supported=True)
    data = mujoco.MjData(model)
    right, left = HandController(model), HandController(model, "left")
    left.command(data, preset(0, "left"))
    if args.viewer:
        from mujoco import viewer as mj_viewer
        commands = queue.SimpleQueue()
        with mj_viewer.launch_passive(model, data, key_callback=commands.put) as viewer:
            viewer.cam.lookat[:] = [.05, -.1, 1.1]
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 180
            count = 1
            while viewer.is_running():
                start = time.monotonic()
                while not commands.empty():
                    key = commands.get()
                    if ord('0') <= key <= ord('3'):
                        count = key-ord('0')
                right.command(data, preset(count))
                for _ in range(20):
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, .02-(time.monotonic()-start)))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(args.output / "a3_supported_hands.xml", encoding="unicode")
    free, free_tree, _ = build_model(supported=False)
    ET.indent(free_tree)
    free_tree.write(args.output / "a3_free_hands.xml", encoding="unicode")
    report = {"hand_model": hand_metadata(), "body_mode": "fixed_kinematic_support", "arm_pose_radians": arm_pose,
              "free_model": {"nq": free.nq, "nv": free.nv, "nu": free.nu}, "poses": []}
    canvas = Image.new("RGB", (640*3, 480*3))
    with mujoco.Renderer(model, 480, 640) as renderer:
        for row, count in enumerate((1, 2, 3)):
            target = preset(count)
            right.command(data, target)
            for _ in range(1500):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            error = right.error(data, target)
            report["poses"].append({"count": count, "max_joint_error_rad": error,
                                    "actual_joint_radians": data.qpos[right.qpos].tolist()})
            print(f"count_{count}: max joint error = {error:.5f} rad", flush=True)
            for col, angle in enumerate((-30, 0, 30)):
                picture = render(renderer, model, data, angle)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 640, 36), fill=(15, 25, 35))
                draw.text((14, 9), f"COUNT {count}  |  audience {angle:+d} deg  |  SUPPORTED", fill="white")
                picture.save(args.output / f"count_{count}_{angle:+d}.png")
                canvas.paste(picture, (col*640, row*480))
            render(renderer, model, data, 0, close=False).save(args.output / f"count_{count}_body.png")
    canvas.save(args.output / "contact_sheet.jpg", quality=95)
    report["finger_tracking_pass"] = all(p["max_joint_error_rad"] < .12 for p in report["poses"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["finger_tracking_pass"]:
        raise SystemExit("Hand tracking failed; inspect contact geometry and report.json")


if __name__ == "__main__":
    main()
