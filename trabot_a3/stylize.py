"""Simulation-side stylisation of Motion Atoms.

A style is a small set of interpretable levers applied to the controlled joints
(waist 3 + arms 14), always relative to the neutral pose so the clip still enters
and leaves through the same bridges:

  gain        amplitude about neutral (arms); waist_gain separately
  tempo       playback-speed multiplier (>1 faster). Implemented as resampling.
  sharpness   >0 accentuates strokes (unsharp mask against a low-pass copy),
              <0 softens them (blend toward the low-pass copy)
  open_deg    shoulder-roll offset outward (+) / inward (-), applied as a smooth
              envelope so the clip's own entry/exit poses are respected
  forward_deg shoulder-pitch offset forward (+)
  lean_deg    waist-pitch lean forward (+)
  hold_s      pause inserted at the strongest stroke apex (calm / emphatic holds)
  mirror      swap left/right (roll and yaw axes negate)

Two ways to choose the levers:
  * presets  — 热情 / 沉稳 / 正式 / 活泼 / 内敛 (named, reviewable, repeatable)
  * reference — measure a reference clip's style descriptors (energy, tempo,
    sharpness) and solve for gain/tempo/sharpness that move the target toward them
    ("参考风格迁移", the motion analogue of a style reference image in video gen)

Every generated variant then goes through the same Eq.10 validation/repair and the
same quality evaluation as recorded atoms; repairs that undo the style are recorded
so a reviewer can see them (flag `style_eroded`).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .data import A3Motion, ALL_JOINTS, CTRL_IDX, ARM_IDX, NEUTRAL_RAD, JOINT_LIMITS_RAD
from .atoms import MotionAtom, CatalogParams, kinematic_report, time_stretch

J = {n: i for i, n in enumerate(ALL_JOINTS)}
WAIST_IDX = np.array([0, 1, 2])


@dataclass
class Style:
    name: str
    gain: float = 1.0
    waist_gain: float = 1.0
    tempo: float = 1.0
    sharpness: float = 0.0
    open_deg: float = 0.0
    forward_deg: float = 0.0
    lean_deg: float = 0.0
    hold_s: float = 0.0
    mirror: bool = False
    zh: str = ""

    def ops(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if k not in ("name", "zh") and v not in (0, 0.0, 1.0, False)}


PRESETS: dict[str, Style] = {
    "enthusiastic": Style("enthusiastic", gain=1.25, waist_gain=1.2, tempo=1.15, sharpness=0.6,
                          open_deg=6, forward_deg=4, lean_deg=3, zh="热情"),
    "calm":         Style("calm", gain=0.85, waist_gain=0.8, tempo=0.82, sharpness=-0.6, hold_s=0.25, zh="沉稳"),
    "formal":       Style("formal", gain=0.7, waist_gain=0.6, tempo=0.95, sharpness=-0.3,
                          forward_deg=3, zh="正式"),
    "lively":       Style("lively", gain=1.15, waist_gain=1.5, tempo=1.3, sharpness=0.4, open_deg=3, zh="活泼"),
    "reserved":     Style("reserved", gain=0.6, waist_gain=0.5, tempo=0.9, sharpness=-0.4,
                          open_deg=-2, zh="内敛"),
}
MIRROR = Style("mirror", mirror=True, zh="左右镜像")

_NEG_ON_MIRROR = {"waist_yaw_joint", "waist_roll_joint", "head_yaw_joint"}
for side in ("left", "right"):
    for jn in ("shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw"):
        _NEG_ON_MIRROR.add(f"{side}_{jn}_joint")


def mirror_q(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i, n in enumerate(ALL_JOINTS):
        src = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        v = q[:, J[src]]
        out[:, i] = -v if n in _NEG_ON_MIRROR else v
    return out


def _envelope(T: int, ramp: int) -> np.ndarray:
    """0 at both ends, 1 in the middle, smooth (quintic) ramps."""
    e = np.ones(T)
    r = max(1, min(ramp, T // 2))
    u = np.linspace(0, 1, r)
    s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    e[:r] = s
    e[-r:] = s[::-1]
    return e


def _apex_frame(q: np.ndarray, fs: float) -> int:
    """Frame of largest arm excursion from neutral (a proxy for the stroke apex)."""
    d = np.linalg.norm(q[:, ARM_IDX] - NEUTRAL_RAD[ARM_IDX], axis=1)
    return int(np.argmax(gaussian_filter1d(d, 2)))


def apply_style(q: np.ndarray, fs: float, st: Style) -> np.ndarray:
    """q: (T,31) rad at fs. Returns the stylised trajectory at the same fs."""
    out = mirror_q(q) if st.mirror else q.copy()
    n = NEUTRAL_RAD
    # amplitude
    out[:, ARM_IDX] = n[ARM_IDX] + st.gain * (out[:, ARM_IDX] - n[ARM_IDX])
    out[:, WAIST_IDX] = n[WAIST_IDX] + st.waist_gain * (out[:, WAIST_IDX] - n[WAIST_IDX])
    # sharpness (controlled joints only), faded at both ends so boundary velocity is untouched
    env = _envelope(out.shape[0], int(0.25 * fs))
    if st.sharpness:
        low = gaussian_filter1d(out[:, CTRL_IDX], sigma=max(1.0, 0.12 * fs), axis=0, mode="nearest")
        w = env[:, None]
        if st.sharpness > 0:
            out[:, CTRL_IDX] = out[:, CTRL_IDX] + w * st.sharpness * (out[:, CTRL_IDX] - low)
        else:
            a = min(1.0, -st.sharpness)
            out[:, CTRL_IDX] = out[:, CTRL_IDX] + w * a * (low - out[:, CTRL_IDX])
    # posture offsets: constant over the clip (a constant offset adds no velocity, so the
    # clip's own entry/exit dynamics are unchanged; the bridge absorbs the shifted entry pose)
    env = np.ones(out.shape[0])
    if st.open_deg:
        o = np.deg2rad(st.open_deg) * env
        out[:, J["left_shoulder_roll_joint"]] += o
        out[:, J["right_shoulder_roll_joint"]] -= o
    if st.forward_deg:
        f = np.deg2rad(st.forward_deg) * env
        out[:, J["left_shoulder_pitch_joint"]] -= f
        out[:, J["right_shoulder_pitch_joint"]] -= f
    if st.lean_deg:
        out[:, J["waist_pitch_joint"]] -= np.deg2rad(st.lean_deg) * env
    # hold at apex
    if st.hold_s > 0:
        k = _apex_frame(out, fs)
        h = int(round(st.hold_s * fs))
        out = np.concatenate([out[:k + 1], np.repeat(out[k:k + 1], h, 0), out[k + 1:]], 0)
    # tempo (resample on a time axis)
    if abs(st.tempo - 1.0) > 1e-6:
        T = out.shape[0]
        n_new = max(4, int(round(T / st.tempo)))
        t_src = np.linspace(0, 1, T)
        t_dst = np.linspace(0, 1, n_new)
        out = np.stack([np.interp(t_dst, t_src, out[:, k]) for k in range(out.shape[1])], 1)
    # keep everything outside the gesture layer on neutral
    keep = np.setdiff1d(np.arange(out.shape[1]), CTRL_IDX)
    out[:, keep] = n[keep]
    return out


# --------------------------------------------------------------------------- reference style
@dataclass
class StyleDescriptor:
    energy: float      # rad, RMS arm excursion from neutral
    tempo: float       # 1/s, mean |qd| / energy
    sharpness: float   # share of arm-velocity power above 2 Hz


def describe(q: np.ndarray, fs: float) -> StyleDescriptor:
    a = q[:, ARM_IDX] - NEUTRAL_RAD[ARM_IDX]
    energy = float(np.sqrt((a ** 2).mean()))
    qd = np.gradient(q[:, ARM_IDX], 1 / fs, axis=0)
    speed = float(np.abs(qd).mean())
    P = np.abs(np.fft.rfft(qd - qd.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(qd.shape[0], 1 / fs)
    sharp = float(P[f > 2.0].sum() / max(P.sum(), 1e-12))
    return StyleDescriptor(energy, speed / max(energy, 1e-3), sharp)


def style_from_reference(target_q: np.ndarray, ref_q: np.ndarray, fs: float, ref_name: str) -> Style:
    """Solve gain/tempo/sharpness so the target moves toward the reference's descriptors."""
    t, r = describe(target_q, fs), describe(ref_q, fs)
    gain = float(np.clip(r.energy / max(t.energy, 1e-3), 0.5, 1.5))
    tempo = float(np.clip(r.tempo / max(t.tempo, 1e-3), 0.6, 1.6))
    sharp = float(np.clip((r.sharpness - t.sharpness) * 4.0, -0.8, 0.8))
    return Style(f"ref:{ref_name}", gain=round(gain, 3), waist_gain=round(gain, 3), tempo=round(tempo, 3),
                 sharpness=round(sharp, 3), zh=f"参考:{ref_name}")


# --------------------------------------------------------------------------- generate + validate
@dataclass
class Variant:
    variant_id: str
    parent_id: str
    style: Style
    q: np.ndarray
    fps: float
    label: str
    hand: str
    repairs: list[str] = field(default_factory=list)
    style_eroded: bool = False

    @property
    def duration(self) -> float:
        return self.q.shape[0] / self.fps


def _hand_of(q: np.ndarray, fs: float, fallback: str) -> str:
    from .atoms import handedness
    return handedness(q[:, ARM_IDX], fs) if q.shape[0] > 3 else fallback


def generate_variant(atom: MotionAtom, st: Style, p: CatalogParams | None = None) -> Variant | None:
    """Stylise, then run the same Eq.10 validation/repair as recorded atoms. None if rejected."""
    p = p or CatalogParams()
    q = apply_style(atom.q, atom.fps, st)
    repairs: list[str] = []
    eroded = False
    ctrl = CTRL_IDX
    lo = JOINT_LIMITS_RAD[ctrl, 0] + p.pos_margin_rad
    hi = JOINT_LIMITS_RAD[ctrl, 1] - p.pos_margin_rad
    over = float(max(0.0, (lo - q[:, ctrl]).max(), (q[:, ctrl] - hi).max()))
    if over > 0:
        # shrink toward neutral just enough, then clip residue
        for k in range(6):
            q[:, ARM_IDX] = NEUTRAL_RAD[ARM_IDX] + 0.9 * (q[:, ARM_IDX] - NEUTRAL_RAD[ARM_IDX])
            q[:, [0, 1, 2]] = NEUTRAL_RAD[[0, 1, 2]] + 0.9 * (q[:, [0, 1, 2]] - NEUTRAL_RAD[[0, 1, 2]])
            over = float(max(0.0, (lo - q[:, ctrl]).max(), (q[:, ctrl] - hi).max()))
            if over <= 0:
                break
        repairs.append(f"缩幅 ×{0.9 ** (k + 1):.2f} 以满足限位裕量")
        eroded = eroded or st.gain > 1.0
        if over > 0:
            return None
    rep = kinematic_report(q, atom.fps, p)
    if rep.stretch > 1.0 + 1e-6:
        if rep.stretch > p.max_stretch or q.shape[0] / atom.fps * rep.stretch > p.max_repaired_s:
            return None
        m = time_stretch(A3Motion("v", atom.fps, np.zeros((q.shape[0], 6)), q), rep.stretch)
        q = m.q
        repairs.append(f"放慢 ×{rep.stretch:.3f}（v {rep.v_max:.1f} / a {rep.a_max:.0f} / j {rep.j_max:.0f}）")
        eroded = eroded or st.tempo > 1.0 or st.sharpness > 0
    vid = f"{atom.atom_id}~{st.name}"
    hand = _hand_of(q, atom.fps, atom.hand)
    return Variant(vid, atom.atom_id, st, q, atom.fps, atom.label, hand, repairs, eroded)
