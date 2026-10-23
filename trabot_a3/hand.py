"""Dexterous hand layer for A3: model, command space, pose library, annotation, stylisation.

Hardware truth (sonic_for_a3 deploy code, `agi_o10_hand.proto` + `o10_hand_bridge.cpp`):
the A3 hand is the AgiBot **O10** — 10 position channels per side, each 0..4096:

    thumb_rotation, thumb_wiggles, thumb_bent,
    index_wiggles, index_bent, middle_bent,
    ring_wiggles, ring_bent, pinky_wiggles, pinky_bent

Simulation model: the AgiBot OmniHand from `agibot_x2_urdf/X2-Ultra_omnihand.urdf` (16 revolute joints
per hand, meshes included), grafted onto the A3 wrists. The 10 O10 channels drive the 16 joints with
the hand's own coupling (pip → dip, thumb mcp → pip → dip), so one library serves both sim and robot.

Normalised pose: 10 floats in [0, 1] per hand ("bend / wiggle fraction"), `to_o10()` → 0..4096.
Which end of a wiggle/rotation channel is 0 on the real hand is an ASSUMPTION to calibrate on hardware
(flagged in `O10_CALIBRATION_NOTE`).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .paths import MJCF, ROOT

O10_CHANNELS = ["thumb_rotation", "thumb_wiggles", "thumb_bent", "index_wiggles", "index_bent",
                "middle_bent", "ring_wiggles", "ring_bent", "pinky_wiggles", "pinky_bent"]
O10_MAX = 4096
O10_CALIBRATION_NOTE = ("bent channels: 0 = straight, 4096 = fully curled (joint at its nonzero limit); "
                        "wiggle/rotation channels: 0..4096 sweeps the URDF joint range low→high (L) / high→low (R) "
                        "so that 0.5 is the neutral abduction. Verify the polarity on hardware before deploying.")
BENT = [2, 4, 5, 7, 9]
WIGGLE = [0, 1, 3, 6, 8]
FINGER_OF = {"thumb": [0, 1, 2], "index": [3, 4], "middle": [5], "ring": [6, 7], "pinky": [8, 9]}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# O10 channel → (URDF joint stem, coupled joints with ratios). Ratios from the URDF limits.
CHANNEL_JOINTS = {
    "thumb_rotation": [("thumb_roll", 1.0)],
    "thumb_wiggles": [("thumb_abad", 1.0)],
    "thumb_bent": [("thumb_mcp", 1.0), ("thumb_pip", 1.1344640 / 0.8552113), ("thumb_dip", 1.1344640 / 0.8552113)],
    "index_wiggles": [("index_abad", 1.0)],
    "index_bent": [("index_pip", 1.0), ("index_dip", 1.8325957 / 1.5707963)],
    "middle_bent": [("middle_pip", 1.0), ("middle_dip", 1.8325957 / 1.5707963)],
    "ring_wiggles": [("ring_abad", 1.0)],
    "ring_bent": [("ring_pip", 1.0), ("ring_dip", 1.8325957 / 1.5707963)],
    "pinky_wiggles": [("pinky_abad", 1.0)],
    "pinky_bent": [("pinky_pip", 1.0), ("pinky_dip", 1.8325957 / 1.5707963)],
}

X2_URDF = Path(os.environ.get("X2_OMNIHAND_URDF",
                              str(ROOT / "external" / "agibot_x2_urdf" / "X2_URDF-v1.4.0" / "X2-Ultra_omnihand.urdf")))


# ----------------------------------------------------------------------------- URDF → MJCF graft
@dataclass
class UJoint:
    name: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray | None
    lo: float
    hi: float
    fixed: bool


def parse_omnihand(urdf: Path = X2_URDF) -> tuple[dict[str, list[UJoint]], dict[str, str]]:
    s = urdf.read_text()
    joints: dict[str, list[UJoint]] = {"L": [], "R": []}
    for n, t, b in re.findall(r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', s, re.S):
        if not re.match(r"[LR]_(palm|thumb|index|middle|ring|pinky)", n):
            continue
        org = re.search(r'<origin[^>]*/>', b).group(0)
        xyz = np.array([float(x) for x in re.search(r'xyz="([^"]+)"', org).group(1).split()])
        m = re.search(r'rpy="([^"]+)"', org)
        rpy = np.array([float(x) for x in m.group(1).split()]) if m else np.zeros(3)
        ax = re.search(r'<axis xyz="([^"]+)"', b)
        lim = re.search(r'<limit[^>]*lower="([^"]+)"[^>]*upper="([^"]+)"', b)
        joints[n[0]].append(UJoint(n, re.search(r'<parent link="([^"]+)"', b).group(1),
                                   re.search(r'<child link="([^"]+)"', b).group(1), xyz, rpy,
                                   np.array([float(x) for x in ax.group(1).split()]) if ax else None,
                                   float(lim.group(1)) if lim else 0.0, float(lim.group(2)) if lim else 0.0, t == "fixed"))
    meshes = {}
    for n, b in re.findall(r'<link name="([^"]+)">(.*?)</link>', s, re.S):
        m = re.search(r'<visual>.*?<mesh filename="([^"]+)"', b, re.S)
        if m:
            meshes[n] = str((urdf.parent / m.group(1)).resolve())
    return joints, meshes


def _quat_wxyz(rpy):
    x, y, z, w = R.from_euler("xyz", rpy).as_quat()
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


# attachment of the OmniHand palm frame to the A3 wrist_yaw frame: URDF +z (fingers) → A3 +x (forearm),
# thumb side → A3 +z (up) in the neutral arm pose; palm faces the body.
ATTACH = {
    "L": {"pos": "0.02 0 0", "rot": np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)},
    "R": {"pos": "0.02 0 0", "rot": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)},
}


def build_a3_hand_mjcf(out_path: Path | None = None, a3_mjcf: str = MJCF, urdf: Path = X2_URDF) -> Path:
    """Write an MJCF of the official A3 with two OmniHands grafted onto the wrists. Returns the path."""
    joints, meshes = parse_omnihand(urdf)
    xml = Path(a3_mjcf).read_text()
    assets, used = [], set()
    bodies = {}
    for side in "LR":
        js = joints[side]
        children: dict[str, list[UJoint]] = {}
        for j in js:
            children.setdefault(j.parent, []).append(j)
        palm = next(j for j in js if j.child.endswith("_palm"))

        def emit(link: str, depth: int) -> str:
            out = []
            for j in children.get(link, []):
                ind = "  " * depth
                q = _quat_wxyz(j.rpy)
                out.append(f'{ind}<body name="{j.child}" pos="{j.xyz[0]:.6f} {j.xyz[1]:.6f} {j.xyz[2]:.6f}" quat="{q}">')
                if not j.fixed:
                    out.append(f'{ind}  <joint name="{j.name}" type="hinge" axis="{j.axis[0]:g} {j.axis[1]:g} {j.axis[2]:g}" '
                               f'range="{j.lo:.6f} {j.hi:.6f}" damping="0.02" armature="0.001"/>')
                if j.child in meshes:
                    mn = Path(meshes[j.child]).stem
                    if mn not in used:
                        used.add(mn); assets.append(f'<mesh name="hand_{mn}" file="{meshes[j.child]}"/>')
                    out.append(f'{ind}  <geom type="mesh" mesh="hand_{mn}" contype="1" conaffinity="1" rgba="0.92 0.92 0.92 1" mass="0.01"/>')
                else:
                    out.append(f'{ind}  <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="0.8 0.3 0.2 0.0" mass="0.001"/>')
                out.append(emit(j.child, depth + 1))
                out.append(f"{ind}</body>")
            return "\n".join(o for o in out if o)
        w, x, y, z = R.from_matrix(ATTACH[side]["rot"]).as_quat()[[3, 0, 1, 2]]
        pal = f'<body name="{palm.child}" pos="{ATTACH[side]["pos"]}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}">\n'
        mn = Path(meshes[palm.child]).stem
        used.add(mn); assets.append(f'<mesh name="hand_{mn}" file="{meshes[palm.child]}"/>')
        pal += f'  <geom type="mesh" mesh="hand_{mn}" contype="1" conaffinity="1" rgba="0.92 0.92 0.92 1" mass="0.15"/>\n'
        pal += emit(palm.child, 1) + "\n</body>"
        bodies[side] = pal
    # inject: after the wrist_yaw_Link's own geom lines (before its closing tag)
    for side, link in (("L", "left_wrist_yaw_Link"), ("R", "right_wrist_yaw_Link")):
        m = re.search(rf'<body name="{link}"[^>]*>', xml)
        assert m, link
        depth, i = 1, m.end()
        while depth:
            o, c = xml.find("<body", i), xml.find("</body>", i)
            if o != -1 and o < c:
                depth += 1; i = o + 5
            else:
                depth -= 1; i = c + 7 if depth else c
        xml = xml[:i] + bodies[side] + "\n" + xml[i:]
    # the official A3 model carries a static cosmetic hand on each wrist link; drop it, the OmniHand replaces it
    xml = re.sub(r'\s*<geom class="(visual|collision)" type="mesh" rgba="[^"]*" mesh="(left|right)_hand_Link"/>', "", xml)
    xml = xml.replace("</asset>", "\n".join(assets) + "\n</asset>", 1)
    out_path = out_path or (ROOT / "results" / "models" / "a3_t2d5_omnihand.xml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # keep A3's relative mesh paths valid: copy the model next to the original meshdir reference
    a3_dir = Path(a3_mjcf).parent
    xml = re.sub(r'meshdir="([^"]+)"', lambda m: f'meshdir="{(a3_dir / m.group(1)).resolve()}"', xml, count=1)
    out_path.write_text(xml)
    return out_path


def hand_joint_names(side: str) -> list[str]:
    """Hinge joints the MJCF carries for one hand, in O10-channel order (coupled joints included)."""
    names = []
    for ch in O10_CHANNELS:
        for stem, _ in CHANNEL_JOINTS[ch]:
            names.append(f"{side}_{stem}_joint")
    return names


class HandModel:
    """Maps normalised O10 poses to MJCF joint angles (and back) for both hands."""

    def __init__(self, urdf: Path = X2_URDF):
        joints, _ = parse_omnihand(urdf)
        self.lim: dict[str, tuple[float, float]] = {}
        for side in "LR":
            for j in joints[side]:
                if not j.fixed:
                    self.lim[j.name] = (j.lo, j.hi)

    def angles(self, side: str, u: np.ndarray) -> dict[str, float]:
        """u: (10,) in [0,1] → {joint name: rad}."""
        out = {}
        for k, ch in enumerate(O10_CHANNELS):
            for stem, ratio in CHANNEL_JOINTS[ch]:
                jn = f"{side}_{stem}_joint"
                lo, hi = self.lim[jn]
                if k in BENT:
                    # bend from 0 toward the nonzero limit; coupled joints (ratio > 1) reach their own limit together
                    end = hi if abs(hi) > abs(lo) else lo
                    out[jn] = float(np.clip(u[k] * end, min(lo, hi), max(lo, hi)))
                else:
                    a, b = (lo, hi) if side == "L" else (hi, lo)
                    out[jn] = float(a + u[k] * (b - a))
        return out

    @staticmethod
    def to_o10(u: np.ndarray) -> list[int]:
        return [int(round(float(np.clip(x, 0, 1)) * O10_MAX)) for x in u]


# ----------------------------------------------------------------------------- pose library
NEUTRAL_HAND = np.array([0.5, 0.5, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25])   # relaxed, slight curl


def pose(thumb_rot=0.5, thumb_wig=0.5, thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0,
         index_wig=0.5, ring_wig=0.5, pinky_wig=0.5) -> np.ndarray:
    return np.array([thumb_rot, thumb_wig, thumb, index_wig, index, middle, ring_wig, ring, pinky_wig, pinky], float)


# Static poses. Chinese one-hand counting 1-10, rock-paper-scissors, and common conversational hand shapes.
POSES: dict[str, dict] = {
    "open":       {"u": pose(0.65, 0.45, 0, 0, 0, 0, 0, 0.6, 0.4, 0.3), "zh": "张开", "spec": {"ext": FINGERS}},
    "relaxed":    {"u": NEUTRAL_HAND, "zh": "放松", "spec": {}},
    "fist":       {"u": pose(0.7, 0.8, 1, 1, 1, 1, 1), "zh": "握拳", "spec": {"curl": FINGERS}},
    "one":        {"u": pose(0.7, 0.8, 1, 0, 1, 1, 1), "zh": "一", "spec": {"ext": ["index"], "curl": ["middle", "ring", "pinky"]}},
    "two":        {"u": pose(0.7, 0.8, 1, 0, 0, 1, 1, 0.7), "zh": "二", "spec": {"ext": ["index", "middle"], "curl": ["ring", "pinky"]}},
    "three":      {"u": pose(0.7, 0.8, 1, 0, 0, 0, 1, 0.65, 0.4), "zh": "三", "spec": {"ext": ["index", "middle", "ring"], "curl": ["pinky"]}},
    "four":       {"u": pose(0.8, 0.9, 1, 0, 0, 0, 0, 0.6, 0.4, 0.35), "zh": "四", "spec": {"ext": ["index", "middle", "ring", "pinky"], "curl": ["thumb"]}},
    "five":       {"u": pose(0.65, 0.4, 0, 0, 0, 0, 0, 0.7, 0.3, 0.2), "zh": "五", "spec": {"ext": FINGERS}},
    "six":        {"u": pose(0.2, 0.1, 0, 1, 1, 1, 0, 0.5, 0.5, 0.15), "zh": "六", "spec": {"ext": ["thumb", "pinky"], "curl": ["index", "middle", "ring"]}},
    "seven":      {"u": pose(0.6, 0.0, 0.6, 0.6, 0.6, 1, 1, 0.5, 0.5, 0.5), "zh": "七", "spec": {"pinch": ["thumb", "index"], "curl": ["ring", "pinky"]}},
    "eight":      {"u": pose(0.65, 0.4, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5), "zh": "八", "spec": {"ext": ["thumb", "index"], "curl": ["middle", "ring", "pinky"]}},
    "nine":       {"u": pose(0.7, 0.8, 1, 0.5, 1, 1, 1, 0.5, 0.5, 0.5), "zh": "九", "spec": {"hook": ["index"], "curl": ["middle", "ring", "pinky"]}},
    "ten":        {"u": pose(0.7, 0.8, 1, 1, 1, 1, 1), "zh": "十", "spec": {"curl": FINGERS}},
    "rock":       {"u": pose(0.7, 0.8, 1, 1, 1, 1, 1), "zh": "石头", "spec": {"curl": FINGERS}},
    "scissors":   {"u": pose(0.7, 0.8, 1, 0, 0, 1, 1, 0.85), "zh": "剪刀", "spec": {"ext": ["index", "middle"], "curl": ["ring", "pinky"], "spread": ["index", "middle"]}},
    "paper":      {"u": pose(0.65, 0.4, 0, 0, 0, 0, 0, 0.75, 0.25, 0.15), "zh": "布", "spec": {"ext": FINGERS}},
    "point":      {"u": pose(0.6, 0.7, 0.8, 0, 1, 1, 1), "zh": "指", "spec": {"ext": ["index"], "curl": ["middle", "ring", "pinky"]}},
    "ok":         {"u": pose(0.6, 0.0, 0.6, 0.6, 0, 0, 0, 0.5, 0.45, 0.4), "zh": "OK", "spec": {"pinch": ["thumb", "index"], "ext": ["middle", "ring", "pinky"]}},
    "thumbs_up":  {"u": pose(0.7, 0.0, 0, 1, 1, 1, 1), "zh": "点赞", "spec": {"ext": ["thumb"], "curl": ["index", "middle", "ring", "pinky"]}},
    "pinch":      {"u": pose(0.6, 0.0, 0.6, 0.6, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5), "zh": "捏", "spec": {"pinch": ["thumb", "index"]}},
    "flat":       {"u": pose(0.7, 0.6, 0.1, 0, 0, 0, 0, 0.5, 0.5, 0.5), "zh": "平掌", "spec": {"ext": FINGERS}},
}
ZH_COUNT = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


# ----------------------------------------------------------------------------- style attributes
@dataclass
class HandStyle:
    """Every attribute is an independent lever; presets are just bundles of them."""
    name: str = "neutral"
    tension: float = 0.0        # curl offset added to bent channels (-0.3 .. +0.3)
    spread: float = 0.0         # abduction gain (-1 = fingers together, +1 = wide)
    tempo: float = 1.0          # time scale of strokes (>1 faster)
    snap: float = 0.0           # 0 = min-jerk, 1 = sharp onset with slight overshoot
    hold: float = 0.0           # seconds added to the hold phase
    anticipation: float = 0.0   # fraction of the stroke amplitude moved the other way first (0..0.3)
    ripple_ms: float = 0.0      # per-finger onset stagger (index → pinky), ms; negative = pinky first
    thumb_lead_ms: float = 0.0  # thumb onset offset vs fingers, ms
    settle: float = 0.0         # residual micro-motion after a hold (0..1), relaxed hands drift a little
    zh: str = "中性"

    def ops(self) -> dict:
        d = asdict(self); d.pop("name"); d.pop("zh")
        return {k: v for k, v in d.items() if v not in (0, 0.0, 1.0)}


HAND_PRESETS = {
    "neutral": HandStyle(),
    "enthusiastic": HandStyle("enthusiastic", tension=-0.05, spread=0.5, tempo=1.3, snap=0.7, hold=0.1, anticipation=0.15, ripple_ms=0, zh="热情"),
    "calm": HandStyle("calm", tension=0.05, spread=-0.2, tempo=0.75, snap=0.0, hold=0.35, ripple_ms=40, settle=0.2, zh="沉稳"),
    "formal": HandStyle("formal", tension=0.0, spread=-0.4, tempo=0.95, snap=0.2, hold=0.2, zh="正式"),
    "lively": HandStyle("lively", tension=-0.1, spread=0.7, tempo=1.45, snap=0.9, anticipation=0.2, ripple_ms=-25, thumb_lead_ms=-40, zh="活泼"),
    "tense": HandStyle("tense", tension=0.25, spread=-0.6, tempo=1.1, snap=0.5, settle=0.0, zh="紧张"),
    "tired": HandStyle("tired", tension=0.15, spread=-0.3, tempo=0.6, snap=0.0, hold=0.5, ripple_ms=70, settle=0.35, zh="疲惫"),
}


def stylise_pose(u: np.ndarray, st: HandStyle) -> np.ndarray:
    v = u.copy()
    v[BENT] = np.clip(v[BENT] + st.tension, 0, 1)
    for k in WIGGLE:
        v[k] = np.clip(0.5 + (v[k] - 0.5) * (1 + st.spread), 0, 1)
    return v


# ----------------------------------------------------------------------------- timed hand atoms
def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def snap_curve(s: np.ndarray, snap: float) -> np.ndarray:
    """Blend between min-jerk and a sharper onset with small overshoot."""
    base = min_jerk(s)
    sharp = 1 - (1 - s) ** 3
    over = 1 + 0.08 * np.sin(np.pi * s) * (s > 0.6)
    return (1 - snap) * base + snap * sharp * over


@dataclass
class HandKey:
    t: float
    u: np.ndarray
    phase: str           # prepare / stroke / hold / release


@dataclass
class HandAtom:
    """a timed hand gesture: label, side, keyframes (seconds from atom start), annotation."""
    atom_id: str
    label: str
    side: str                           # "L" | "R"
    keys: list[HandKey]
    style: HandStyle = field(default_factory=HandStyle)
    annotation: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.keys[-1].t

    def sample(self, t: float) -> np.ndarray:
        """pose at local time t with per-finger ripple / thumb lead and the style's onset curve."""
        ks = self.keys
        if t <= ks[0].t:
            return ks[0].u.copy()
        if t >= ks[-1].t:
            return ks[-1].u.copy()
        out = np.zeros(10)
        for k in range(10):
            f = next(f for f, chans in FINGER_OF.items() if k in chans)
            off = (st := self.style).thumb_lead_ms / 1000 if f == "thumb" else st.ripple_ms / 1000 * (FINGERS.index(f) - 1)
            tk = t - off
            i = max(0, min(len(ks) - 2, int(np.searchsorted([x.t for x in ks], tk, side="right") - 1)))
            a, b = ks[i], ks[i + 1]
            s = float(np.clip((tk - a.t) / max(b.t - a.t, 1e-6), 0, 1))
            w = snap_curve(np.array([s]), st.snap)[0] if b.phase == "stroke" else min_jerk(np.array([s]))[0]
            out[k] = a.u[k] + w * (b.u[k] - a.u[k])
        return np.clip(out, 0, 1)

    def frames(self, fps: float = 30.0) -> np.ndarray:
        n = int(round(self.duration * fps)) + 1
        return np.stack([self.sample(i / fps) for i in range(n)])

    def meta(self) -> dict:
        return {"atom_id": self.atom_id, "label": self.label, "side": self.side, "duration": round(self.duration, 3),
                "keys": [{"t": round(k.t, 3), "phase": k.phase, "o10": HandModel.to_o10(k.u)} for k in self.keys],
                "style": self.style.name, "style_ops": self.style.ops(), "annotation": self.annotation}


def make_atom(label: str, side: str, start: np.ndarray | None = None, st: HandStyle | None = None,
              stroke_s: float = 0.28, hold_s: float = 0.45, release: bool = True, atom_id: str | None = None) -> HandAtom:
    """Build prepare → stroke → hold → (release) keyframes for a library pose, applying every style lever."""
    st = st or HandStyle()
    start = NEUTRAL_HAND.copy() if start is None else start.copy()
    target = stylise_pose(POSES[label]["u"], st)
    stroke = stroke_s / st.tempo
    hold = hold_s + st.hold
    keys = [HandKey(0.0, start, "prepare")]
    t = 0.0
    if st.anticipation > 0:
        anti = np.clip(start - st.anticipation * (target - start), 0, 1)
        t += 0.35 * stroke; keys.append(HandKey(t, anti, "prepare"))
    t += stroke; keys.append(HandKey(t, target, "stroke"))
    t += hold
    settled = target if st.settle == 0 else np.clip(target + st.settle * 0.06 * (NEUTRAL_HAND - target), 0, 1)
    keys.append(HandKey(t, settled, "hold"))
    if release:
        t += stroke * 1.2; keys.append(HandKey(t, NEUTRAL_HAND.copy(), "release"))
    ann = {"label": label, "label_zh": POSES[label]["zh"], "side": side, "spec": POSES[label]["spec"],
           "phases": [{"name": k.phase, "t": round(k.t, 3)} for k in keys],
           "contacts": ["thumb-index" if "pinch" in POSES[label]["spec"] else "none"],
           "style": st.name, "style_ops": st.ops(), "source": "procedural", "reviewed": False}
    return HandAtom(atom_id or f"{label}_{side}_{st.name}", label, side, keys, st, ann)
