"""Direction-matching retargeting of a humanoid skeleton into the 17 A3 gesture joints.

Why directions: the public datasets use different skeletons, rest poses, units and up axes.
Bone *directions* expressed in the performer's own chest frame are invariant to all of that,
and the A3 arm has the same topology (3-DoF shoulder, 1-DoF elbow, 3-DoF wrist), so each arm
is solved as a small bounded least-squares problem per frame:

    minimise  |dir(upper arm) - d_u|² + |dir(forearm) - d_f|² + 0.5 |dir(hand) - d_h|²  + λ|q - q_prev|²
    subject to A3 joint limits

with A3 forward kinematics taken straight from the official MJCF (body offsets, body quats, joint
axes). Wrist roll is unobservable from directions alone and is regularised to its neutral value.
The waist is set from the lean of the spine and the twist of the shoulders relative to the hips.

Fidelity is reported per frame as the angular error of the three directions (degrees) — it is
one of the benchmark metrics, because a dataset whose gestures cannot be reproduced on A3's
kinematics is not a good source for the library whatever its other merits.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from trabot_a3.data import ALL_JOINTS, NEUTRAL_RAD, JOINT_LIMITS_RAD
from trabot_a3.paths import MJCF
from .bvh import BVH
from .skeleton import resolve

ARM_CHAIN = {
    "L": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
          "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "R": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
          "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
}
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
HAND_END_LOCAL = np.array([0.08, 0.0, 0.0])   # palm/finger direction in the wrist_yaw frame


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


class A3ArmFK:
    """Numpy FK of one A3 arm chain, torso frame → shoulder, elbow, wrist, hand-end positions."""

    def __init__(self, side: str):
        m = mujoco.MjModel.from_xml_path(MJCF)
        self.names = ARM_CHAIN[side]
        self.pos, self.quat, self.axis = [], [], []
        for n in self.names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            b = m.jnt_bodyid[j]
            w, x, y, z = m.body_quat[b]
            self.pos.append(m.body_pos[b].copy())
            self.quat.append(R.from_quat([x, y, z, w]))
            self.axis.append(m.jnt_axis[j].copy())
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_Link")
        self.torso_pos = m.body_pos[torso].copy()
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]
        self.lo = np.array([m.jnt_range[j, 0] for j in jid]) + 0.02
        self.hi = np.array([m.jnt_range[j, 1] for j in jid]) - 0.02
        self.q_idx = np.array([ALL_JOINTS.index(n) for n in self.names])
        self.q_neutral = NEUTRAL_RAD[self.q_idx]

    def points(self, q: np.ndarray) -> np.ndarray:
        """(4,3): shoulder (pitch joint origin), elbow, wrist (pitch joint origin), hand end — in torso frame."""
        p = np.zeros(3)
        rot = R.identity()
        out = []
        for k in range(7):
            p = p + rot.apply(self.pos[k])
            rot = rot * self.quat[k] * R.from_rotvec(self.axis[k] * q[k])
            if k in (0, 3, 5):
                out.append(p.copy())
        out.append(p + rot.apply(HAND_END_LOCAL))
        return np.array(out)

    def dirs(self, q: np.ndarray) -> np.ndarray:
        P = self.points(q)
        return _unit(np.stack([P[1] - P[0], P[2] - P[1], P[3] - P[2]]))

    def solve(self, d_target: np.ndarray, q0: np.ndarray, lam: float = 0.05, hand: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """hand=False: no hand-end landmark (e.g. a zero-length End Site) → wrist pitch/yaw stay near neutral."""
        w = np.array([1.0, 1.0, 0.7 if hand else 0.0])

        def res(q):
            d = self.dirs(q)
            e = ((d - d_target) * w[:, None]).ravel()
            reg = lam * (q - q0)
            reg[4] += 0.2 * (q[4] - self.q_neutral[4])     # wrist roll: unobservable → stay near neutral
            if not hand:
                reg[5] += 0.2 * (q[5] - self.q_neutral[5]); reg[6] += 0.2 * (q[6] - self.q_neutral[6])
            return np.concatenate([e, reg])
        q0c = np.clip(q0, self.lo, self.hi)
        r = least_squares(res, q0c, bounds=(self.lo, self.hi), xtol=1e-5, ftol=1e-6, max_nfev=60)
        d = self.dirs(r.x)
        ang = np.degrees(np.arccos(np.clip((d * d_target).sum(1), -1, 1)))
        return r.x, ang


@dataclass
class RetargetResult:
    q: np.ndarray             # (T,31) rad, A3 joints at the source fps (legs/head neutral)
    fps: float
    dir_err_deg: np.ndarray   # (T,2,3) per side: upper arm, forearm, hand
    landmarks: dict[str, str]

    @property
    def fidelity(self) -> dict:
        e = self.dir_err_deg
        return {"mean_deg": round(float(e.mean()), 2), "p95_deg": round(float(np.percentile(e, 95)), 2),
                "upper_arm_deg": round(float(e[:, :, 0].mean()), 2), "forearm_deg": round(float(e[:, :, 1].mean()), 2),
                "hand_deg": round(float(e[:, :, 2].mean()), 2)}


def chest_frames(P: dict[str, np.ndarray]) -> np.ndarray:
    """(T,3,3) rotation whose rows are the performer's forward / left / up axes in world coords."""
    left = _unit(P["l_shoulder"] - P["r_shoulder"])
    up = _unit(P["neck"] - P["hips"])
    up = _unit(up - (up * left).sum(1, keepdims=True) * left)
    fwd = np.cross(left, up)
    return np.stack([fwd, left, up], 1)


def retarget_landmarks(P: dict[str, np.ndarray], fps: float, up_axis: np.ndarray = np.array([0, 1.0, 0]),
                       landmarks: dict[str, str] | None = None) -> RetargetResult:
    """P: landmark name → (T,3) world positions (any unit). Returns A3 joint angles."""
    T = P["hips"].shape[0]
    F = chest_frames(P)                     # (T,3,3), rows = axes
    q = np.repeat(NEUTRAL_RAD[None], T, 0)
    errs = np.zeros((T, 2, 3))
    fks = {"L": A3ArmFK("L"), "R": A3ArmFK("R")}
    for si, side in enumerate("LR"):
        s = side.lower()
        d_u = _unit(P[f"{s}_elbow"] - P[f"{s}_shoulder"])
        d_f = _unit(P[f"{s}_wrist"] - P[f"{s}_elbow"])
        hv = P[f"{s}_hand_end"] - P[f"{s}_wrist"]
        has_hand = np.linalg.norm(hv, axis=1) > 1e-6
        d_h = _unit(hv)
        fk = fks[side]
        qprev = fk.q_neutral.copy()
        for t in range(T):
            tgt = np.stack([F[t] @ d_u[t], F[t] @ d_f[t], F[t] @ d_h[t] if has_hand[t] else F[t] @ d_f[t]])
            sol, ang = fk.solve(tgt, qprev, hand=bool(has_hand[t]))
            if not has_hand[t]:
                ang[2] = 0.0
            q[t, fk.q_idx] = sol
            errs[t, si] = ang
            qprev = sol
    # waist: spine lean relative to world up, twist relative to hips heading
    spine = _unit(P["neck"] - P["hips"])
    left = _unit(P["l_shoulder"] - P["r_shoulder"])
    left_h = _unit(left - (left * up_axis).sum(1, keepdims=True) * up_axis)
    fwd_h = np.cross(left_h, up_axis)
    f_c = (spine * fwd_h).sum(1); l_c = (spine * left_h).sum(1); u_c = (spine * up_axis).sum(1)
    pitch = -np.arctan2(f_c, u_c)            # A3: positive waist pitch leans back
    roll = -np.arctan2(l_c, u_c)             # A3: positive waist roll tilts right
    if "l_hip" in P and "r_hip" in P:
        hip_left = _unit(P["l_hip"] - P["r_hip"])
        hip_left = _unit(hip_left - (hip_left * up_axis).sum(1, keepdims=True) * up_axis)
        yaw = np.arctan2(np.cross(hip_left, left_h) @ up_axis, (hip_left * left_h).sum(1))
    else:
        yaw = np.zeros(T)
    for name, val in zip(WAIST, [yaw, roll, pitch]):
        i = ALL_JOINTS.index(name)
        q[:, i] = np.clip(val, JOINT_LIMITS_RAD[i, 0] + 0.02, JOINT_LIMITS_RAD[i, 1] - 0.02)
    return RetargetResult(q, fps, errs, landmarks or {})


def retarget_bvh(bvh: BVH, profile: str | dict | None = None, up_axis=np.array([0, 1.0, 0]),
                 target_fps: float = 30.0) -> RetargetResult:
    """Solve at `target_fps` (gesture content lives well below 15 Hz; the solver is the cost)."""
    names = [j.name for j in bvh.joints]
    lm = resolve(names, profile)
    pos, _ = bvh.fk()
    stride = max(1, int(round(bvh.fps / target_fps)))
    pos = pos[::stride]
    fps = bvh.fps / stride
    P = {k: pos[:, bvh.index(v)] for k, v in lm.items()}
    # optional hip landmarks for waist yaw
    for key, pats in (("l_hip", ("LeftUpLeg", "b_l_upleg", "LeftHip", "UpperLeg_L")), ("r_hip", ("RightUpLeg", "b_r_upleg", "RightHip", "UpperLeg_R"))):
        for p in pats:
            if p in names:
                P[key] = pos[:, bvh.index(p)]
                break
    return retarget_landmarks(P, fps, up_axis=up_axis, landmarks=lm)
