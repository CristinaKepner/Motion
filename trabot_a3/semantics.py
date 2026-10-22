"""Checkable action specifications for the 17 communicative functions.

"动作名对了，动作却没做对": a label (from a file name, a VLM, or a generator prompt)
is not evidence that the motion performs the function. Following the MotionCoach idea
(arXiv:2610.12963 — decompose a motion name into checkable pose, contact and temporal
order, and verify candidate references before using them), every function here is a
short list of kinematic checks on the A3 keypoints:

  pose      where the hands are relative to shoulders / chest / head
  contact   proximity relations (hand-to-chest, hand-to-hand separation)
  temporal  ordered phases (raise → strike → stop, extend → retract, approach → hold),
            beat counts, oscillation counts, holds

Each check reports pass/fail, the measured value, and the time window that satisfied it,
so a reviewer sees *which part* of the action is missing. Specs are deliberately simple
and conservative; they are a self-check and a retrieval filter, not a classifier, and the
thresholds are a first draft meant to be tuned against human review.

Coordinates: metres, robot faces +x, +y is the robot's left, +z up. Keypoints as in
quality.KEYPOINT_BODIES.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .data import ALL_JOINTS

PELVIS, TORSO, HEAD, SL, EL, HL, SR, ER, HR = 0, 1, 2, 3, 4, 5, 6, 7, 8
WAIST_PITCH = ALL_JOINTS.index("waist_pitch_joint")

FN_ZH = {"emphasize": "强调", "enumerate": "列举", "contrast_compare": "对比", "refer_self": "指自己",
         "refer_other": "指对方", "refer_external": "指外部", "depict_size_shape": "描绘大小形状",
         "depict_direction_motion": "描绘方向运动", "turn_open": "开启话轮", "turn_yield": "让出话轮",
         "greet_farewell": "问候告别", "thank": "感谢", "affirm": "肯定", "deny": "否定",
         "certainty": "确定", "uncertainty": "不确定", "enthusiasm": "热情"}


@dataclass
class Check:
    name: str            # 中文，可读
    kind: str            # pose / contact / temporal
    passed: bool
    value: str           # measured value, human-readable
    need: str            # requirement, human-readable
    t: tuple[float, float] | None = None


@dataclass
class SpecResult:
    function: str
    passed: bool
    score: float
    checks: list[Check] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [asdict(c) for c in self.checks]
        return d


class Feat:
    """Shared kinematic features of one clip."""

    def __init__(self, kp: np.ndarray, q: np.ndarray, fs: float):
        self.fs = fs
        self.T = kp.shape[0]
        self.t = np.arange(self.T) / fs
        k = gaussian_filter1d(kp, 1.0, axis=0, mode="nearest") if self.T > 4 else kp
        self.k = k
        self.H = {"L": k[:, HL], "R": k[:, HR]}
        self.S = {"L": k[:, SL], "R": k[:, SR]}
        sh_mid = 0.5 * (k[:, SL] + k[:, SR])
        self.chest = sh_mid + np.array([0.08, 0.0, -0.18])
        self.head = k[:, HEAD]
        self.shoulder_w = float(np.linalg.norm(k[0, SL] - k[0, SR]))
        self.V = {s: np.gradient(self.H[s], 1 / fs, axis=0) for s in "LR"}
        self.speed = {s: np.linalg.norm(self.V[s], axis=1) for s in "LR"}
        self.path = {s: float(self.speed[s].sum() / fs) for s in "LR"}
        self.dom = "L" if self.path["L"] >= self.path["R"] else "R"
        self.out = {"L": 1.0, "R": -1.0}            # sign of "outward" on y
        self.waist_pitch = q[:, WAIST_PITCH]
        self.sep = np.linalg.norm(self.H["L"] - self.H["R"], axis=1)

    def win(self, mask: np.ndarray) -> tuple[float, float] | None:
        idx = np.nonzero(mask)[0]
        return (round(float(idx[0] / self.fs), 2), round(float(idx[-1] / self.fs), 2)) if len(idx) else None

    def peaks(self, s: str, min_h: float = 0.35, prom: float = 0.15):
        sp = self.speed[s]
        p, _ = find_peaks(sp, height=min_h, prominence=prom, distance=max(1, int(0.2 * self.fs)))
        return p

    def hold_after(self, s: str, i: int, thr: float = 0.15, need_s: float = 0.15) -> float:
        """seconds the hand stays slow after frame i."""
        sp = self.speed[s]
        n = 0
        for j in range(i, self.T):
            if sp[j] < thr:
                n += 1
            elif n:
                break
        return n / self.fs

    def osc(self, s: str, axis: int = 1, amp: float = 0.04) -> int:
        """number of direction reversals of hand position along axis with ≥ amp travel."""
        x = self.H[s][:, axis]
        rev, ext, d = 0, x[0], 0
        for v in x[1:]:
            if d >= 0 and v > ext:
                ext, d = v, 1 if v - x[0] > 0 else d
            elif d <= 0 and v < ext:
                ext, d = v, -1 if v - x[0] < 0 else d
            if d == 1 and ext - v >= amp:
                rev, ext, d = rev + 1, v, -1
            elif d == -1 and v - ext >= amp:
                rev, ext, d = rev + 1, v, 1
        return rev


def _c(name, kind, ok, value, need, t=None) -> Check:
    return Check(name, kind, bool(ok), value, need, t)


def _fwd(f: Feat, s: str) -> np.ndarray:
    return f.H[s][:, 0] - f.S[s][:, 0]


def _up(f: Feat, s: str) -> np.ndarray:
    return f.H[s][:, 2] - f.S[s][:, 2]


def _lat(f: Feat, s: str) -> np.ndarray:
    return (f.H[s][:, 1] - f.S[s][:, 1]) * f.out[s]


def _reach(f: Feat, s: str) -> np.ndarray:
    return np.linalg.norm(f.H[s] - f.S[s], axis=1)


# ------------------------------------------------------------------ the 17 specs
def spec_emphasize(f: Feat) -> list[Check]:
    s = f.dom
    p = f.peaks(s, 0.6, 0.25)
    out = [_c("有明显击打（手速峰 ≥ 0.6 m/s）", "temporal", len(p) > 0,
              f"峰值 {f.speed[s].max():.2f} m/s", "≥ 0.6 m/s",
              f.win(np.isin(np.arange(f.T), p)) if len(p) else None)]
    stop_ok, tw = False, None
    for i in p:
        j = min(f.T - 1, i + int(0.25 * f.fs))
        if f.speed[s][i:j + 1].min() < 0.4 * f.speed[s][i]:
            stop_ok, tw = True, (round(i / f.fs, 2), round(j / f.fs, 2))
            break
    out.append(_c("击打后 0.25 s 内急停（降到峰值 40% 以下）", "temporal", stop_ok, "满足" if stop_ok else "未急停", "击打→急停", tw))
    hi = _up(f, s)[p].max() if len(p) else _up(f, s).max()
    out.append(_c("击打时手在胸口高度附近（肩下 ≤ 0.45 m）", "pose", hi > -0.45, f"相对肩 {hi:+.2f} m", "> −0.45 m"))
    return out


def spec_enumerate(f: Feat) -> list[Check]:
    s = f.dom
    p = f.peaks(s, 0.35, 0.12)
    gaps = np.diff(p) / f.fs if len(p) >= 2 else np.array([])
    cv = float(gaps.std() / gaps.mean()) if len(gaps) >= 2 else (0.0 if len(gaps) == 1 else 9.0)
    front = _fwd(f, s)[p].mean() if len(p) else _fwd(f, s).mean()
    return [_c("至少 2 个节拍（速度峰）", "temporal", len(p) >= 2, f"{len(p)} 个", "≥ 2"),
            _c("节拍间隔规律（变异系数 < 0.5）", "temporal", len(p) >= 2 and cv < 0.5, f"CV {cv:.2f}" if len(gaps) else "—", "< 0.5"),
            _c("节拍发生在身前（手在肩前 ≥ 0.10 m）", "pose", front >= 0.10, f"{front:+.2f} m", "≥ 0.10 m")]


def spec_contrast(f: Feat) -> list[Check]:
    w = int(0.4 * f.fs)
    L = np.convolve(f.speed["L"], np.ones(w) / w, "same")
    R = np.convolve(f.speed["R"], np.ones(w) / w, "same")
    domL = (L > 1.6 * R) & (L > 0.2)
    domR = (R > 1.6 * L) & (R > 0.2)
    order = None
    if domL.any() and domR.any():
        order = "左→右" if np.argmax(domL) < np.argmax(domR) else "右→左"
    return [_c("先后出现左手主导段和右手主导段", "temporal", order is not None, order or "只有一侧主导", "左右交替"),
            _c("两只手各自在身体同侧展开（外展 ≥ 0.10 m）", "pose",
               _lat(f, "L").max() >= 0.10 and _lat(f, "R").max() >= 0.10,
               f"L {_lat(f, 'L').max():+.2f} / R {_lat(f, 'R').max():+.2f} m", "两侧都 ≥ 0.10 m")]


def spec_refer_self(f: Feat) -> list[Check]:
    d = {s: np.linalg.norm(f.H[s] - f.chest, axis=1) for s in "LR"}
    s = "L" if d["L"].min() < d["R"].min() else "R"
    i = int(np.argmin(d[s]))
    approach = d[s][0] - d[s][i]
    return [_c("手贴近胸口（距胸 < 0.18 m）", "contact", d[s].min() < 0.18, f"{d[s].min():.2f} m", "< 0.18 m",
               f.win(d[s] < 0.18)),
            _c("先靠近后停住（靠近 ≥ 0.10 m，再停留 ≥ 0.15 s）", "temporal",
               approach >= 0.10 and f.hold_after(s, i) >= 0.15, f"靠近 {approach:.2f} m，停 {f.hold_after(s, i):.2f} s", "靠近→停留")]


def spec_refer_other(f: Feat) -> list[Check]:
    s = f.dom
    fw = _fwd(f, s)
    i = int(np.argmax(fw))
    return [_c("手向前伸出（肩前 ≥ 0.30 m）", "pose", fw.max() >= 0.30, f"{fw.max():.2f} m", "≥ 0.30 m", f.win(fw >= 0.30)),
            _c("手臂基本伸直（手到肩 ≥ 0.45 m）", "pose", _reach(f, s)[i] >= 0.45, f"{_reach(f, s)[i]:.2f} m", "≥ 0.45 m"),
            _c("伸到最远处停留 ≥ 0.15 s", "temporal", f.hold_after(s, max(0, i - 2)) >= 0.15, f"{f.hold_after(s, max(0, i - 2)):.2f} s", "≥ 0.15 s")]


def spec_refer_external(f: Feat) -> list[Check]:
    s = f.dom
    ext = np.maximum(_fwd(f, s), _lat(f, s))
    i = int(np.argmax(ext))
    return [_c("手伸向身体外侧或前方（≥ 0.25 m）", "pose", ext.max() >= 0.25, f"{ext.max():.2f} m", "≥ 0.25 m", f.win(ext >= 0.25)),
            _c("手臂基本伸直（手到肩 ≥ 0.45 m）", "pose", _reach(f, s)[i] >= 0.45, f"{_reach(f, s)[i]:.2f} m", "≥ 0.45 m"),
            _c("伸出后停留 ≥ 0.15 s", "temporal", f.hold_after(s, max(0, i - 2)) >= 0.15, f"{f.hold_after(s, max(0, i - 2)):.2f} s", "≥ 0.15 s")]


def spec_size_shape(f: Feat) -> list[Check]:
    both = min(f.path.values()) >= 0.15
    vL, vR = f.V["L"], f.V["R"] * np.array([1, -1, 1])   # mirror right hand onto left
    num = (vL * vR).sum()
    den = np.sqrt((vL ** 2).sum() * (vR ** 2).sum()) + 1e-9
    sym = float(num / den)
    rng = float(np.ptp(f.sep))
    return [_c("双手都参与（各自路径 ≥ 0.15 m）", "pose", both, f"L {f.path['L']:.2f} / R {f.path['R']:.2f} m", "都 ≥ 0.15 m"),
            _c("双手镜像对称运动（相关 ≥ 0.5）", "temporal", sym >= 0.5, f"{sym:.2f}", "≥ 0.5"),
            _c("双手间距明显变化（≥ 0.15 m），勾勒出大小", "contact", rng >= 0.15, f"{rng:.2f} m", "≥ 0.15 m")]


def spec_direction(f: Feat) -> list[Check]:
    s = f.dom
    net = float(np.linalg.norm(f.H[s][-1] - f.H[s][0]))
    seg = f.H[s]
    best, tw = 0.0, None
    w = int(0.8 * f.fs)
    for a in range(0, max(1, f.T - w), max(1, w // 4)):
        b = min(f.T - 1, a + w)
        d = float(np.linalg.norm(seg[b] - seg[a]))
        if d > best:
            best, tw = d, (round(a / f.fs, 2), round(b / f.fs, 2))
    straight = best / max(1e-6, float(f.speed[s][int(tw[0] * f.fs):int(tw[1] * f.fs) + 1].sum() / f.fs)) if tw else 0.0
    return [_c("0.8 s 内单向扫过 ≥ 0.25 m", "temporal", best >= 0.25, f"{best:.2f} m", "≥ 0.25 m", tw),
            _c("扫动路径较直（位移/路程 ≥ 0.6）", "pose", straight >= 0.6, f"{straight:.2f}", "≥ 0.6"),
            _c("不是原路往返（首末位置相距 ≥ 0.10 m 或只扫一次）", "temporal", net >= 0.10 or f.osc(s, 1, 0.1) <= 1, f"首末 {net:.2f} m", "单向")]


def spec_turn_open(f: Feat) -> list[Check]:
    i0 = int(np.argmin(f.sep[: max(2, f.T // 2)]))
    gain = float(f.sep[i0:].max() - f.sep[i0])
    fw_end = min(_fwd(f, "L")[-max(1, f.T // 4):].mean(), _fwd(f, "R")[-max(1, f.T // 4):].mean())
    return [_c("双手由内向外打开（间距增大 ≥ 0.15 m）", "temporal", gain >= 0.15, f"+{gain:.2f} m", "≥ 0.15 m",
               (round(i0 / f.fs, 2), round((i0 + int(np.argmax(f.sep[i0:]))) / f.fs, 2))),
            _c("打开后双手在身前（肩前 ≥ 0.05 m）", "pose", fw_end >= 0.05, f"{fw_end:+.2f} m", "≥ 0.05 m")]


def spec_turn_yield(f: Feat) -> list[Check]:
    s = f.dom
    fw = _fwd(f, s)
    i = int(np.argmax(fw))
    back = float(fw[i] - fw[-1])
    return [_c("单手向前递出（肩前 ≥ 0.25 m）", "pose", fw.max() >= 0.25, f"{fw.max():.2f} m", "≥ 0.25 m"),
            _c("递出后收回（回收 ≥ 0.15 m）", "temporal", back >= 0.15 and i < f.T - 3, f"回收 {back:.2f} m", "递出→收回",
               (round(i / f.fs, 2), round((f.T - 1) / f.fs, 2)))]


def spec_greet(f: Feat) -> list[Check]:
    s = "L" if _up(f, "L").max() > _up(f, "R").max() else "R"
    up = _up(f, s)
    high = up > 0.05
    n = f.osc(s, 1, 0.04) if high.any() else 0
    return [_c("手举过肩（肩上 ≥ 0.05 m）", "pose", high.any(), f"{up.max():+.2f} m", "≥ +0.05 m", f.win(high)),
            _c("举高后左右摆动 ≥ 2 次", "temporal", n >= 2, f"{n} 次", "≥ 2")]


def spec_thank(f: Feat) -> list[Check]:
    d = np.maximum(np.linalg.norm(f.H["L"] - f.chest, axis=1), np.linalg.norm(f.H["R"] - f.chest, axis=1))
    bow = float(np.rad2deg(-(f.waist_pitch - f.waist_pitch[0]).min()))
    return [_c("双手同时靠近胸前（都距胸 < 0.25 m）", "contact", d.min() < 0.25, f"{d.min():.2f} m", "< 0.25 m", f.win(d < 0.25)),
            _c("身体前倾致意（腰部前俯 ≥ 8°）", "pose", bow >= 8, f"{bow:.1f}°", "≥ 8°")]


def spec_affirm(f: Feat) -> list[Check]:
    s = f.dom
    vz = f.V[s][:, 2]
    p, _ = find_peaks(-vz, height=0.25, prominence=0.1, distance=max(1, int(0.2 * f.fs)))
    ext = float(np.ptp(f.H[s], axis=0).max())
    return [_c("1–3 个向下的小节拍", "temporal", 1 <= len(p) <= 3, f"{len(p)} 个", "1–3"),
            _c("幅度小（手活动范围 < 0.25 m）", "pose", ext < 0.25, f"{ext:.2f} m", "< 0.25 m")]


def spec_deny(f: Feat) -> list[Check]:
    s = f.dom
    front = _fwd(f, s) >= 0.08
    n = f.osc(s, 1, 0.04)
    return [_c("手在身前（肩前 ≥ 0.08 m）", "pose", front.any(), f"{_fwd(f, s).max():+.2f} m", "≥ 0.08 m", f.win(front)),
            _c("左右摆手 ≥ 2 次", "temporal", n >= 2, f"{n} 次", "≥ 2")]


def spec_certainty(f: Feat) -> list[Check]:
    s = f.dom
    vz = f.V[s][:, 2]
    i = int(np.argmin(vz))
    return [_c("有力向下切（下行速度 ≥ 0.5 m/s）", "temporal", -vz[i] >= 0.5, f"{-vz[i]:.2f} m/s", "≥ 0.5 m/s"),
            _c("下切后定住 ≥ 0.2 s", "temporal", f.hold_after(s, i, 0.15) >= 0.2, f"{f.hold_after(s, i, 0.15):.2f} s", "≥ 0.2 s",
               (round(i / f.fs, 2), round(min(f.T - 1, i + int(0.2 * f.fs)) / f.fs, 2)))]


def spec_uncertainty(f: Feat) -> list[Check]:
    spread = f.sep.max() - f.shoulder_w
    fw = min(_fwd(f, "L").max(), _fwd(f, "R").max())
    peak = max(f.speed["L"].max(), f.speed["R"].max())
    return [_c("双手摊开（间距超过肩宽 ≥ 0.15 m）", "pose", spread >= 0.15, f"+{spread:.2f} m", "≥ 0.15 m", f.win(f.sep - f.shoulder_w >= 0.15)),
            _c("双手在身前（肩前 ≥ 0.10 m）", "pose", fw >= 0.10, f"{fw:+.2f} m", "≥ 0.10 m"),
            _c("动作缓慢（手速峰 < 0.8 m/s）", "temporal", peak < 0.8, f"{peak:.2f} m/s", "< 0.8 m/s")]


def spec_enthusiasm(f: Feat) -> list[Check]:
    up = max(_up(f, "L").max(), _up(f, "R").max())
    ext = max(float(np.ptp(f.H[s], axis=0).max()) for s in "LR")
    peak = max(f.speed["L"].max(), f.speed["R"].max())
    return [_c("手举到肩上方（≥ +0.10 m）", "pose", up >= 0.10, f"{up:+.2f} m", "≥ +0.10 m"),
            _c("动作幅度大（活动范围 ≥ 0.35 m）", "pose", ext >= 0.35, f"{ext:.2f} m", "≥ 0.35 m"),
            _c("速度快（手速峰 ≥ 0.9 m/s）", "temporal", peak >= 0.9, f"{peak:.2f} m/s", "≥ 0.9 m/s")]


SPECS = {
    "emphasize": spec_emphasize, "enumerate": spec_enumerate, "contrast_compare": spec_contrast,
    "refer_self": spec_refer_self, "refer_other": spec_refer_other, "refer_external": spec_refer_external,
    "depict_size_shape": spec_size_shape, "depict_direction_motion": spec_direction,
    "turn_open": spec_turn_open, "turn_yield": spec_turn_yield, "greet_farewell": spec_greet,
    "thank": spec_thank, "affirm": spec_affirm, "deny": spec_deny,
    "certainty": spec_certainty, "uncertainty": spec_uncertainty, "enthusiasm": spec_enthusiasm,
}

# a short written description of each spec, for prompts and the review page
SPEC_TEXT = {
    "emphasize": "主导手在胸口高度快速击打（≥0.6 m/s），随后 0.25 s 内急停",
    "enumerate": "主导手在身前做 ≥2 个节奏均匀的节拍",
    "contrast_compare": "先一侧手主导、再另一侧手主导，两手各在同侧展开",
    "refer_self": "手移向并停在自己胸口（<0.18 m）",
    "refer_other": "手臂伸直向正前方伸出 ≥0.30 m 并短暂停住",
    "refer_external": "手臂伸直伸向外侧或前方 ≥0.25 m 并短暂停住",
    "depict_size_shape": "双手镜像对称运动，双手间距变化 ≥0.15 m",
    "depict_direction_motion": "主导手在 0.8 s 内沿较直路径单向扫过 ≥0.25 m",
    "turn_open": "双手由内向外打开（间距 +0.15 m），停在身前",
    "turn_yield": "单手向前递出 ≥0.25 m，然后收回",
    "greet_farewell": "手举过肩并左右摆动 ≥2 次",
    "thank": "双手同时到胸前，身体前倾 ≥8°",
    "affirm": "1–3 个向下小节拍，活动范围 <0.25 m",
    "deny": "手在身前左右摆动 ≥2 次",
    "certainty": "有力向下切（≥0.5 m/s）后定住 ≥0.2 s",
    "uncertainty": "双手在身前摊开，动作缓慢",
    "enthusiasm": "手举过肩，幅度大（≥0.35 m）且速度快",
}


def check(function: str, kp: np.ndarray, q: np.ndarray, fs: float, feat: Feat | None = None) -> SpecResult:
    f = feat or Feat(kp, q, fs)
    cs = SPECS[function](f)
    score = sum(c.passed for c in cs) / len(cs)
    return SpecResult(function, all(c.passed for c in cs), round(score, 3), cs)


def check_all(kp: np.ndarray, q: np.ndarray, fs: float) -> dict[str, SpecResult]:
    f = Feat(kp, q, fs)
    return {fn: check(fn, kp, q, fs, f) for fn in SPECS}
