"""Minimal, dependency-free BVH reader with forward kinematics.

Handles the conventions of the public gesture datasets (BEAT, ZEGGS, Trinity, Talking With
Hands / GENEA, HiPHI): arbitrary channel orders (Zrotation Xrotation Yrotation etc.), root
translation, End Sites, any linear unit (returned as-is; the caller rescales).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]
    children: list[int] = field(default_factory=list)
    is_end: bool = False


@dataclass
class BVH:
    joints: list[Joint]
    frame_time: float
    values: np.ndarray            # (T, n_channels)
    chan_index: list[tuple[int, str]]  # per channel: (joint index, channel name)

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def T(self) -> int:
        return self.values.shape[0]

    def names(self) -> list[str]:
        return [j.name for j in self.joints if not j.is_end]

    def index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(name)

    def local_rotations(self) -> np.ndarray:
        """(T, J) scipy Rotation per joint (as quaternions xyzw array (T,J,4))."""
        T, J = self.T, len(self.joints)
        quats = np.zeros((T, J, 4))
        quats[:, :, 3] = 1.0
        per_joint: dict[int, list[tuple[str, int]]] = {}
        for ci, (ji, ch) in enumerate(self.chan_index):
            if ch.endswith("rotation"):
                per_joint.setdefault(ji, []).append((ch[0].upper(), ci))
        for ji, chans in per_joint.items():
            seq = "".join(ax for ax, _ in chans)              # e.g. "ZXY" — BVH order is intrinsic
            angles = np.stack([self.values[:, ci] for _, ci in chans], 1)
            quats[:, ji] = R.from_euler(seq, angles, degrees=True).as_quat()
        return quats

    def fk(self) -> tuple[np.ndarray, np.ndarray]:
        """World positions (T,J,3) and world rotations as quaternions xyzw (T,J,4)."""
        T, J = self.T, len(self.joints)
        lq = self.local_rotations()
        pos = np.zeros((T, J, 3))
        rot = np.zeros((T, J, 4))
        # local translation: a joint WITH position channels animates its translation (the channel
        # holds the full local offset, as MotionBuilder/Vicon exports do); otherwise the static OFFSET.
        trans = np.repeat(np.stack([j.offset for j in self.joints])[None], T, 0)
        has_pos = set()
        for ci, (ji, ch) in enumerate(self.chan_index):
            if ch.endswith("position"):
                trans[:, ji, "XYZ".index(ch[0].upper())] = self.values[:, ci]
                has_pos.add(ji)
        for ji, j in enumerate(self.joints):
            if j.parent < 0:
                rot[:, ji] = lq[:, ji]
                pos[:, ji] = trans[:, ji] if ji in has_pos else j.offset
            else:
                pr = R.from_quat(rot[:, j.parent])
                pos[:, ji] = pos[:, j.parent] + pr.apply(trans[:, ji])
                rot[:, ji] = (pr * R.from_quat(lq[:, ji])).as_quat()
        return pos, rot


def load_bvh(path: str) -> BVH:
    with open(path) as f:
        tokens = f.read().split()
    i = 0
    joints: list[Joint] = []
    chan_index: list[tuple[int, str]] = []
    stack: list[int] = []

    def expect(tok):
        nonlocal i
        if tokens[i].upper() != tok:
            raise ValueError(f"BVH parse error: expected {tok}, got {tokens[i]} at token {i}")
        i += 1

    expect("HIERARCHY")
    while tokens[i].upper() != "MOTION":
        t = tokens[i].upper()
        if t in ("ROOT", "JOINT"):
            name = tokens[i + 1]; i += 2
            expect("{")
            parent = stack[-1] if stack else -1
            joints.append(Joint(name, parent, np.zeros(3), []))
            jid = len(joints) - 1
            if parent >= 0:
                joints[parent].children.append(jid)
            stack.append(jid)
        elif t == "END":
            i += 2  # "End Site"
            expect("{")
            parent = stack[-1]
            joints.append(Joint(joints[parent].name + "_End", parent, np.zeros(3), [], is_end=True))
            jid = len(joints) - 1
            joints[parent].children.append(jid)
            stack.append(jid)
        elif t == "OFFSET":
            joints[stack[-1]].offset = np.array([float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            chans = tokens[i + 2: i + 2 + n]
            joints[stack[-1]].channels = chans
            for c in chans:
                chan_index.append((stack[-1], c))
            i += 2 + n
        elif t == "}":
            stack.pop(); i += 1
        else:
            raise ValueError(f"BVH parse error at token {i}: {tokens[i]}")
    expect("MOTION")
    expect("FRAMES:")
    n_frames = int(tokens[i]); i += 1
    expect("FRAME")
    expect("TIME:")
    frame_time = float(tokens[i]); i += 1
    n_ch = len(chan_index)
    vals = np.array(tokens[i: i + n_frames * n_ch], dtype=np.float64).reshape(n_frames, n_ch)
    return BVH(joints, frame_time, vals, chan_index)


def write_bvh(path: str, joints: list[Joint], frame_time: float, values: np.ndarray, chan_index: list[tuple[int, str]]):
    """Write a BVH (used for the synthetic round-trip test)."""
    lines = ["HIERARCHY"]

    def emit(ji: int, depth: int):
        j = joints[ji]
        ind = "  " * depth
        if j.is_end:
            lines.append(f"{ind}End Site"); lines.append(f"{ind}{{")
            lines.append(f"{ind}  OFFSET {j.offset[0]:.6f} {j.offset[1]:.6f} {j.offset[2]:.6f}")
            lines.append(f"{ind}}}"); return
        lines.append(f"{ind}{'ROOT' if j.parent < 0 else 'JOINT'} {j.name}"); lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {j.offset[0]:.6f} {j.offset[1]:.6f} {j.offset[2]:.6f}")
        lines.append(f"{ind}  CHANNELS {len(j.channels)} {' '.join(j.channels)}")
        for c in j.children:
            emit(c, depth + 1)
        lines.append(f"{ind}}}")
    emit(0, 0)
    lines += ["MOTION", f"Frames: {values.shape[0]}", f"Frame Time: {frame_time:.6f}"]
    lines += [" ".join(f"{v:.6f}" for v in row) for row in values]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
