"""Self-checks for the adapters (no external data needed).

1. A3 → landmarks → retarget → A3: joint-angle and direction errors on real catalog atoms.
2. Synthetic Mixamo-style BVH with analytic poses: parser + FK + retarget give the expected A3 angles.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trabot_a3.atoms import load_catalog  # noqa: E402
from trabot_a3.data import ALL_JOINTS, NEUTRAL_RAD, NEUTRAL_ROOT, CTRL_IDX  # noqa: E402
from trabot_a3.paths import MJCF, OUT  # noqa: E402
from benchmark.bvh import Joint, write_bvh, load_bvh  # noqa: E402
from benchmark.retarget import retarget_landmarks, retarget_bvh, HAND_END_LOCAL  # noqa: E402

BODIES = {"hips": "pelvis_link", "chest": "torso_Link", "neck": "head_yaw_Link",
          "l_shoulder": "left_shoulder_pitch_Link", "l_elbow": "left_elbow_Link", "l_wrist": "left_wrist_pitch_Link",
          "r_shoulder": "right_shoulder_pitch_Link", "r_elbow": "right_elbow_Link", "r_wrist": "right_wrist_pitch_Link",
          "l_hip": "left_hip_pitch_Link", "r_hip": "right_hip_pitch_Link"}


def a3_landmarks(q: np.ndarray) -> dict[str, np.ndarray]:
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ALL_JOINTS]
    bid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, v) for k, v in BODIES.items()}
    wy = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_Link") for s in ("left", "right")}
    P = {k: np.zeros((q.shape[0], 3)) for k in list(BODIES) + ["l_hand_end", "r_hand_end"]}
    for t in range(q.shape[0]):
        d.qpos[:3] = NEUTRAL_ROOT[:3]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[qadr] = q[t]
        mujoco.mj_kinematics(m, d)
        for k, b in bid.items():
            P[k][t] = d.xpos[b]
        for s, key in (("left", "l_hand_end"), ("right", "r_hand_end")):
            P[key][t] = d.xpos[wy[s]] + d.xmat[wy[s]].reshape(3, 3) @ HAND_END_LOCAL
    return P


def test_a3_roundtrip(n_atoms: int = 6) -> dict:
    cat = load_catalog(OUT / "catalog")
    picks = [a for a in cat.atoms if a.source.startswith(("080_reach", "077_reach", "058_punch", "063_twist"))][:n_atoms]
    rows = []
    t0 = time.time()
    for a in picks:
        q = a.q[::3]                              # 10 Hz is enough for a check
        P = a3_landmarks(q)
        res = retarget_landmarks(P, 10.0, up_axis=np.array([0, 0, 1.0]))
        obs = [i for i in CTRL_IDX if "wrist_roll" not in ALL_JOINTS[i] and "waist_yaw" not in ALL_JOINTS[i]]
        err = np.degrees(np.abs(res.q[:, obs] - q[:, obs]))
        rows.append({"atom": a.atom_id, "frames": int(q.shape[0]), "joint_err_mean_deg": round(float(err.mean()), 2),
                     "joint_err_p95_deg": round(float(np.percentile(err, 95)), 2), **res.fidelity})
    return {"atoms": rows, "seconds": round(time.time() - t0, 1)}


def synthetic_bvh(path: str, poses: list[dict[str, tuple[float, float, float]]], fps: float = 30.0):
    """Mixamo/BEAT-like upper body, Y-up, cm, channels 'Zrotation Xrotation Yrotation'. Arms rest along -Y (A-pose down)."""
    chans = ["Zrotation", "Xrotation", "Yrotation"]
    J = []
    def add(name, parent, off, end=False):
        J.append(Joint(name, parent, np.array(off, float), [] if end else chans, [], end));
        if parent >= 0: J[parent].children.append(len(J) - 1)
        return len(J) - 1
    hips = add("Hips", -1, (0, 95, 0)); J[hips].channels = ["Xposition", "Yposition", "Zposition"] + chans
    sp = add("Spine", hips, (0, 10, 0)); sp1 = add("Spine1", sp, (0, 10, 0)); sp2 = add("Spine2", sp1, (0, 10, 0)); sp3 = add("Spine3", sp2, (0, 10, 0))
    neck = add("Neck", sp3, (0, 12, 0)); head = add("Head", neck, (0, 10, 0)); add("Head_End", head, (0, 10, 0), True)
    for side, sgn in (("Left", 1), ("Right", -1)):
        sh = add(f"{side}Shoulder", sp3, (sgn * 5, 8, 0))
        arm = add(f"{side}Arm", sh, (sgn * 12, 0, 0))
        fa = add(f"{side}ForeArm", arm, (0, -28, 0))
        hd = add(f"{side}Hand", fa, (0, -25, 0))
        mid = add(f"{side}HandMiddle1", hd, (0, -8, 0)); add(f"{side}HandMiddle1_End", mid, (0, -4, 0), True)
    for side, sgn in (("Left", 1), ("Right", -1)):
        ul = add(f"{side}UpLeg", hips, (sgn * 9, -2, 0)); lg = add(f"{side}Leg", ul, (0, -42, 0)); ft = add(f"{side}Foot", lg, (0, -42, 0)); add(f"{side}Foot_End", ft, (0, -5, 10), True)
    chan_index = [(ji, c) for ji, j in enumerate(J) for c in j.channels]
    T = len(poses)
    vals = np.zeros((T, len(chan_index)))
    for t, pose in enumerate(poses):
        for ci, (ji, c) in enumerate(chan_index):
            if c == "Yposition" and J[ji].parent < 0:
                vals[t, ci] = 0.0
            r = pose.get(J[ji].name)
            if r and c.endswith("rotation"):
                vals[t, ci] = r["ZXY".index(c[0])]
    write_bvh(path, J, 1.0 / fps, vals, chan_index)


def test_bvh_parser() -> dict:
    # arms rest pointing down (-Y). Rotating LeftArm +90° about Z (ZXY order) lifts the left arm sideways to +X = outward.
    poses = [dict(), {"LeftArm": (90, 0, 0)}, {"LeftArm": (90, 0, 0), "LeftForeArm": (0, -90, 0)},
             {"RightArm": (-90, 0, 0)}, {"Spine1": (0, 20, 0)}]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "synthetic.bvh")
        synthetic_bvh(p, poses)
        b = load_bvh(p)
        pos, _ = b.fk()
        la, lf, lh = (b.index(n) for n in ("LeftArm", "LeftForeArm", "LeftHand"))
        rest_dir = pos[0, lf] - pos[0, la]; up_dir = pos[1, lf] - pos[1, la]
        res = retarget_bvh(b, "beat")
    deg = lambda i: round(float(np.degrees(res.q[i][ALL_JOINTS.index(nm)])), 1)
    out = {"frames": b.T, "fps": b.fps, "joints": len(b.joints), "landmarks": res.landmarks,
           "rest_upper_arm_dir": rest_dir.round(1).tolist(), "raised_upper_arm_dir": up_dir.round(1).tolist(),
           "fidelity": res.fidelity}
    nm = "left_shoulder_roll_joint"; out["left_shoulder_roll_deg_rest/raised"] = [deg(0), deg(1)]
    nm = "left_elbow_joint"; out["left_elbow_deg_rest/raised/bent"] = [deg(0), deg(1), deg(2)]
    nm = "right_shoulder_roll_joint"; out["right_shoulder_roll_deg_rest/raised"] = [deg(0), deg(3)]
    nm = "waist_pitch_joint"; out["waist_pitch_deg_rest/lean"] = [deg(0), deg(4)]
    return out


if __name__ == "__main__":
    r1 = test_bvh_parser()
    print(json.dumps(r1, ensure_ascii=False, indent=1))
    r2 = test_a3_roundtrip()
    print(json.dumps(r2, ensure_ascii=False, indent=1))
    (OUT / "benchmark").mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark" / "adapter_selfcheck.json").write_text(json.dumps({"bvh_parser": r1, "a3_roundtrip": r2}, ensure_ascii=False, indent=1))
