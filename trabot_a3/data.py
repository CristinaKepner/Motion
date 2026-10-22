"""A3 flat-CSV loading and joint conventions.

CSV convention (sonic_for_a3/a3_data/agibot_a3): 120 Hz, 38 columns,
root_translate in cm, root_rotate/joints in degrees. Joint column names match
the official A3 MJCF joint names one-to-one.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_COLS = ["root_translateX", "root_translateY", "root_translateZ",
             "root_rotateX", "root_rotateY", "root_rotateZ"]

ALL_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# TRABot controls 14 upper-limb joints (G1). On A3 the direct analogue is the
# 7-DoF left arm + 7-DoF right arm. Waist/head/legs stay on their neutral pose
# (paper: fixed root, arms only) or come from the whole-body controller.
ARM_JOINTS = ALL_JOINTS[5:19]
LEFT_ARM = ALL_JOINTS[5:12]
RIGHT_ARM = ALL_JOINTS[12:19]
ARM_IDX = np.array([ALL_JOINTS.index(j) for j in ARM_JOINTS])
LEFT_ARM_IDX = np.array([ALL_JOINTS.index(j) for j in LEFT_ARM])
RIGHT_ARM_IDX = np.array([ALL_JOINTS.index(j) for j in RIGHT_ARM])
# Joints the gesture layer commands: waist (3) + both arms (14). Head and legs stay
# on the neutral pose (paper: fixed root, upper-limb control only).
CTRL_IDX = np.array([0, 1, 2] + list(ARM_IDX))

# Neutral (recovery) pose: the official MJCF "stand" keyframe, in degrees.
NEUTRAL_DEG = np.array([
    0.0, 0.01, 0.78,              # waist
    0.0, 0.0,                     # head (keyframe has -9.7 yaw; we centre it)
    16.92, 6.42, -0.16, 46.25, 0.01, 0.45, -0.04,     # left arm
    16.92, -6.42, 0.16, 46.26, -0.01, 0.46, 0.05,     # right arm
    -7.54, 0.46, -2.02, 14.41, -7.41, -0.55,          # left leg
    -7.53, -0.43, 2.01, 14.41, -7.41, 0.58,           # right leg
])
NEUTRAL_RAD = np.deg2rad(NEUTRAL_DEG)
NEUTRAL_ROOT = np.array([-0.042, 0.0, 1.07, 0.0, 0.0, 0.0])  # x y z (m), euler xyz (deg)

# Joint position limits (rad) from the official A3 MJCF, same order as ALL_JOINTS.
JOINT_LIMITS_RAD = np.array([
    [-2.62, 2.62], [-0.35, 0.35], [-0.49, 0.42],
    [-1.05, 1.05], [-0.44, 0.26],
    [-2.88, 2.88], [-0.09, 2.62], [-2.79, 2.79], [-0.96, 1.75], [-2.79, 2.79], [-1.62, 1.62], [-1.62, 1.62],
    [-2.88, 2.88], [-2.62, 0.09], [-2.79, 2.79], [-0.96, 1.75], [-2.79, 2.79], [-1.62, 1.62], [-1.62, 1.62],
    [-2.51, 2.93], [-0.52, 1.61], [-2.72, 2.72], [-0.12, 2.5], [-0.91, 0.52], [-0.35, 0.35],
    [-2.51, 2.93], [-1.61, 0.52], [-2.72, 2.72], [-0.12, 2.5], [-0.91, 0.52], [-0.35, 0.35],
])


@dataclass
class A3Motion:
    """One recording: root pose + 31 joint angles (rad), at `fps`."""
    name: str
    fps: float
    root: np.ndarray      # (T, 6)  x y z in m, euler xyz in deg
    q: np.ndarray         # (T, 31) rad

    @property
    def T(self) -> int:
        return self.q.shape[0]

    @property
    def duration(self) -> float:
        return self.T / self.fps

    def resample(self, fps: float) -> "A3Motion":
        """Linear resampling on a true time axis (paper: map by timestamp, not frame index)."""
        if abs(fps - self.fps) < 1e-9:
            return self
        t_src = np.arange(self.T) / self.fps
        n = int(np.floor(t_src[-1] * fps)) + 1
        t_dst = np.arange(n) / fps
        root = np.stack([np.interp(t_dst, t_src, self.root[:, k]) for k in range(6)], 1)
        q = np.stack([np.interp(t_dst, t_src, self.q[:, k]) for k in range(self.q.shape[1])], 1)
        return A3Motion(self.name, fps, root, q)

    def slice(self, a: int, b: int, name: str | None = None) -> "A3Motion":
        return A3Motion(name or self.name, self.fps, self.root[a:b].copy(), self.q[a:b].copy())


def load_a3_csv(path: str | Path, fps: float = 120.0) -> A3Motion:
    path = Path(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV {path}")
    root = np.array([[float(r[c]) for c in ROOT_COLS] for r in rows], dtype=np.float64)
    root[:, :3] *= 0.01
    q_deg = np.array([[float(r[j]) for j in ALL_JOINTS] for r in rows], dtype=np.float64)
    if not np.all(np.isfinite(q_deg)):
        raise ValueError(f"{path} contains non-finite joint values")
    return A3Motion(path.stem, fps, root, np.deg2rad(q_deg))


def write_a3_csv(path: str | Path, motion: A3Motion) -> None:
    """Write a 38-column flat CSV accepted by sonic_for_a3 sim2sim (--csv-source-fps motion.fps)."""
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame", *ROOT_COLS, *ALL_JOINTS])
        q_deg = np.rad2deg(motion.q)
        for i in range(motion.T):
            root = motion.root[i].copy()
            root[:3] *= 100.0
            w.writerow([i, *[f"{v:.6f}" for v in root], *[f"{v:.6f}" for v in q_deg[i]]])
