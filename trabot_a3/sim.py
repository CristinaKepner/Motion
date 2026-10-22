"""MuJoCo kinematic playback backend on the official A3 MJCF.

Joint angles are written straight into qpos (the "把关节角输进去" path from the
project notes). An optional first-order tracking lag emulates a whole-body
controller that does not reach the commanded pose instantly, so the executor's
re-planning actually sees a measured pose that differs from the catalog's nominal
exit pose (the paper's "0.2 rad planned vs 0.5 rad measured" situation).

Also records the full 31-joint trajectory that was *commanded* and the one that
was *measured*, for CSV export and the trace.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import mujoco  # noqa: E402

from .data import ALL_JOINTS, CTRL_IDX, NEUTRAL_RAD, NEUTRAL_ROOT
from .executor import SimBackend


@dataclass
class FrameRecord:
    tag: str
    q_cmd: np.ndarray
    q_meas: np.ndarray
    t: float


class MujocoKinematicBackend(SimBackend):
    def __init__(self, mjcf_path: str, fps: float = 30.0, tracking_tau: float = 0.0,
                 render: bool = True, width: int = 960, height: int = 540, cam_distance: float = 2.3):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.data = mujoco.MjData(self.model)
        self.fps = fps
        self.tau = tracking_tau
        self.qadr = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                              for j in ALL_JOINTS])
        self.q_meas = NEUTRAL_RAD.copy()
        self.t = 0.0
        self.records: list[FrameRecord] = []
        self.frames: list[np.ndarray] = []
        self.render = render
        if render:
            self.renderer = mujoco.Renderer(self.model, height, width)
            self.cam = mujoco.MjvCamera()
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = [0.0, 0.0, 0.95]
            self.cam.distance = cam_distance
            self.cam.azimuth = 150
            self.cam.elevation = -12
        self._write(NEUTRAL_RAD)

    def _write(self, q_full: np.ndarray):
        d = self.data
        d.qpos[0:3] = NEUTRAL_ROOT[:3]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = q_full
        mujoco.mj_forward(self.model, d)

    def read_arm_state(self) -> np.ndarray:
        return self.q_meas[CTRL_IDX].copy()

    def play(self, q_full: np.ndarray, tag: str) -> tuple[float, np.ndarray]:
        dt = 1.0 / self.fps
        alpha = 1.0 if self.tau <= 0 else 1.0 - np.exp(-dt / self.tau)
        for i in range(q_full.shape[0]):
            cmd = q_full[i]
            self.q_meas = self.q_meas + alpha * (cmd - self.q_meas)
            self._write(self.q_meas)
            self.records.append(FrameRecord(tag, cmd.copy(), self.q_meas.copy(), self.t))
            if self.render:
                self.renderer.update_scene(self.data, camera=self.cam)
                self.frames.append(self.renderer.render().copy())
            self.t += dt
        return q_full.shape[0] * dt, self.q_meas[CTRL_IDX].copy()

    # ---- self-contact check used by the offline validator (paper: reject contacts >2 mm deeper than neutral)
    def contact_depth(self, q_full: np.ndarray) -> float:
        self._write(q_full)
        mujoco.mj_collision(self.model, self.data)
        worst = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]; b2 = self.model.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:   # ignore ground
                continue
            worst = max(worst, -c.dist)
        return worst

    def commanded_trajectory(self) -> np.ndarray:
        return np.stack([r.q_cmd for r in self.records]) if self.records else np.zeros((0, 31))

    def measured_trajectory(self) -> np.ndarray:
        return np.stack([r.q_meas for r in self.records]) if self.records else np.zeros((0, 31))
