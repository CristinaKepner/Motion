"""Offscreen rendering helpers for the A3 + OmniHand model (MUJOCO_GL=osmesa by default)."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .data import ALL_JOINTS, NEUTRAL_RAD  # noqa: E402
from .hand import HandModel, build_a3_hand_mjcf, hand_joint_names  # noqa: E402
from .paths import ROOT  # noqa: E402

MODEL_PATH = ROOT / "results" / "models" / "a3_t2d5_omnihand.xml"


class HandScene:
    def __init__(self, w: int = 960, h: int = 720):
        if not MODEL_PATH.exists():
            build_a3_hand_mjcf(MODEL_PATH)
        self.m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.d = mujoco.MjData(self.m)
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = max(w, 1280), max(h, 960)
        self.r = mujoco.Renderer(self.m, h, w)
        self.hm = HandModel()
        self.qadr = {self.m.joint(i).name: self.m.jnt_qposadr[i] for i in range(self.m.njnt)}
        self.body_idx = {n: i for i, n in enumerate(ALL_JOINTS)}
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.set_body(NEUTRAL_RAD)

    def set_body(self, q31: np.ndarray):
        for n, v in zip(ALL_JOINTS, q31):
            self.d.qpos[self.qadr[n]] = v

    def set_hand(self, side: str, u: np.ndarray):
        for n, v in self.hm.angles(side, u).items():
            self.d.qpos[self.qadr[n]] = v

    def forward(self):
        mujoco.mj_forward(self.m, self.d)

    def fingertips(self, side: str) -> dict[str, np.ndarray]:
        return {f: self.d.body(f"{side}_{f}_tip").xpos.copy() for f in ["thumb", "index", "middle", "ring", "pinky"]}

    def palm(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        return self.d.body(f"{side}_palm").xpos.copy(), self.d.body(f"{side}_palm").xmat.reshape(3, 3).copy()

    def render(self, lookat=None, azimuth=180, elevation=-15, distance=1.2) -> np.ndarray:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat if lookat is not None else (self.d.body("L_palm").xpos + self.d.body("R_palm").xpos) / 2
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        self.r.update_scene(self.d, cam)
        return self.r.render().copy()

    def render_hand(self, side: str, distance=0.35, azimuth=None, elevation=-25) -> np.ndarray:
        p, _ = self.palm(side)
        tips = self.fingertips(side)
        look = (p + tips["middle"]) / 2
        az = azimuth if azimuth is not None else (150 if side == "L" else 210)
        return self.render(look, az, elevation, distance)


# ----------------------------------------------------------------------------- 名实核对 for hands (FK-based)
EXT_Z, CURL_Z, PINCH_M, SPREAD_M = 0.15, 0.11, 0.035, 0.03


def check_hand_spec(scene: HandScene, side: str, u: np.ndarray, spec: dict) -> dict:
    """Check a pose against its label's spec using fingertip positions in the palm frame.
    ext: tip z > EXT_Z (finger extended) · curl: tip z < CURL_Z · pinch: tip–tip distance < PINCH_M ·
    spread: neighbouring extended tips > SPREAD_M apart · hook: index between ext and curl."""
    scene.set_hand(side, u)
    scene.forward()
    p, Rm = scene.palm(side)
    tips = {f: Rm.T @ (v - p) for f, v in scene.fingertips(side).items()}
    checks = {}
    for f in spec.get("ext", []):
        checks[f"ext:{f}"] = (bool(tips[f][2] > EXT_Z) if f != "thumb" else bool(np.linalg.norm(tips[f]) > 0.11), round(float(tips[f][2]), 3))
    for f in spec.get("curl", []):
        checks[f"curl:{f}"] = (bool(tips[f][2] < CURL_Z) if f != "thumb" else bool(tips[f][0] < 0.03), round(float(tips[f][2]), 3))
    pair = spec.get("pinch")
    if pair:
        d = float(np.linalg.norm(tips[pair[0]] - tips[pair[1]]))
        checks[f"pinch:{pair[0]}-{pair[1]}"] = (d < PINCH_M, round(d, 3))
    sp = spec.get("spread")
    if sp:
        d = float(np.linalg.norm(tips[sp[0]] - tips[sp[1]]))
        checks[f"spread:{sp[0]}-{sp[1]}"] = (d > SPREAD_M, round(d, 3))
    for f in spec.get("hook", []):
        checks[f"hook:{f}"] = (bool(CURL_Z < tips[f][2] < EXT_Z + 0.02 and tips[f][0] > 0.02), round(float(tips[f][2]), 3))
    return {"pass": all(v[0] for v in checks.values()), "checks": checks,
            "tips_palm_frame": {f: np.round(v, 4).tolist() for f, v in tips.items()}}
