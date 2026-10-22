"""Motion-library quality & richness evaluation.

Per clip (atom or generated variant), four sub-scores, each 0-100:

  自然度 naturalness   SPARC smoothness of hand speed, high-frequency jitter share (>5 Hz),
                       jerk vs limit
  连贯性 coherence     single-frame jumps (spikes), entry/exit velocity (a clip that starts or
                       ends while moving produces a visible jolt against the zero-velocity
                       quintic bridges), bridge cost from neutral
  可执行 executability joint-limit margin, self-collision depth beyond the neutral baseline,
                       fixed-root inverse-dynamics torque / actuatorfrcrange, PD tracking error
                       of a torque-limited fixed-root forward simulation
  表现力 expressiveness hand path length / workspace extent / peak speed; penalises clips that
                       are near-static or rhythmic arm swing (gait-like periodicity)

Every penalty carries a human-readable flag so a reviewer sees *why* a clip scored low.
Library-level: label coverage of the 17 functions, per-label diversity, near-duplicates,
hand balance, style-space coverage (energy × tempo), connectivity from neutral.

Physics here is a proxy for the whole-body controller: fixed root, gravity on, the official
A3 torque limits. The SONIC sim2sim metrics JSON can be merged in via `merge_controller_metrics`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402

from .data import ALL_JOINTS, CTRL_IDX, ARM_IDX, NEUTRAL_RAD, NEUTRAL_ROOT, JOINT_LIMITS_RAD
from .atoms import SEMANTIC_FUNCTIONS
from .bridge import bridge_duration

KEYPOINT_BODIES = [
    "pelvis_link", "torso_Link", "head_pitch_Link",
    "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
    "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link",
    "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
    "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link",
]
HAND_L, HAND_R = 5, 8

# A3 joint PD gains / armature, copied from sonic_for_a3 sim2sim (get_a3_joint_control,
# utils/a3_motor_params.py) so the tracking proxy uses the same servo layer the policy sees.
_ARM = {"78_58": 30.118e-6 * 20.03 ** 2, "93_65": 138.5069e-6 * 21.906 ** 2,
        "41_48": 1.359e-6 * 24.415 ** 2, "59_60": 10.1374e-6 * 22.136 ** 2, "110_75": 300.851e-6 * 20.0 ** 2}


def a3_joint_gains(name: str) -> tuple[float, float, float]:
    """(kp, kd, armature) for one A3 joint."""
    if "_hip_roll" in name: return 120, 4, _ARM["93_65"]
    if "_hip_yaw" in name or "_hip_pitch" in name: return 80, 3, _ARM["93_65"]
    if "_knee" in name: return 250, 8, _ARM["110_75"]
    if "_ankle_pitch" in name: return 50, 2, _ARM["78_58"] * 5.333
    if "_ankle_roll" in name: return 50, 2, _ARM["78_58"] * 1.66562
    if name == "waist_yaw_joint": return 85, 3, _ARM["93_65"]
    if name == "waist_roll_joint": return 50, 2, _ARM["78_58"] * 1.21
    if name == "waist_pitch_joint": return 50, 2, _ARM["78_58"] * 7.3
    if "_shoulder_pitch" in name or "_shoulder_roll" in name: return 40, 3, _ARM["78_58"]
    if "_shoulder_yaw" in name or "_elbow" in name or "_wrist_roll" in name: return 30, 2, _ARM["59_60"]
    if "_wrist_pitch" in name or "_wrist_yaw" in name: return 20, 2, _ARM["41_48"]
    if "head" in name: return 20, 1, _ARM["41_48"]
    raise KeyError(name)
SHOULDER_L, SHOULDER_R = 3, 6

FLAG_TEXT = {
    "jitter": "高频抖动（>5 Hz 能量占比高）",
    "unsmooth": "手部速度曲线不平滑（SPARC 低）",
    "jerk": "jerk 接近或超过限值",
    "spike": "单帧跳变（数据断点或重定向错误）",
    "entry_vel": "起始时仍在运动，桥接后会顿挫",
    "exit_vel": "结束时仍在运动，衔接/归位会顿挫",
    "far_entry": "入口姿态离中立远，接入成本高",
    "limit_margin": "关节离限位不足 0.05 rad",
    "self_collision": "自碰撞比中立姿态深 >2 mm",
    "torque": "所需力矩超过电机限值",
    "torque_near": "所需力矩接近电机限值（>80%）",
    "tracking": "受力矩约束的跟踪误差大",
    "static": "几乎不动，表达信息少",
    "periodic": "节律性摆臂（像步态带出的，不像手势）",
    "too_short": "时长过短",
    "style_eroded": "风格被安全修复削弱（放慢/缩幅）",
    "semantic_mismatch": "动作没有通过所贴语义标签的规格（名对、动作不对）",
}
SEVERITY = {"spike": 3, "torque": 3, "self_collision": 3, "limit_margin": 2, "jitter": 2, "entry_vel": 2,
            "exit_vel": 2, "tracking": 2, "unsmooth": 1, "jerk": 1, "far_entry": 1, "torque_near": 1,
            "static": 2, "periodic": 1, "too_short": 1, "style_eroded": 1, "semantic_mismatch": 2}


@dataclass
class ClipQuality:
    clip_id: str
    score: float
    grade: str
    sub: dict[str, float]
    metrics: dict[str, float]
    flags: list[str] = field(default_factory=list)

    def flag_text(self) -> list[str]:
        return [FLAG_TEXT[f] for f in self.flags]


# ------------------------------------------------------------------ smoothness
def sparc(speed: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length (Balasubramanian et al. 2015). ~-1.4 very smooth, < -3 jerky."""
    if speed.max() < 1e-6:
        return -1.4
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    Mf = Mf / Mf.max()
    sel = f <= fc
    f_sel, Mf_sel = f[sel], Mf[sel]
    idx = np.nonzero(Mf_sel >= amp_th)[0]
    f_sel, Mf_sel = f_sel[idx[0]:idx[-1] + 1], Mf_sel[idx[0]:idx[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0] + 1e-12)
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(Mf_sel) ** 2)))


def hf_rms(qd: np.ndarray, fs: float, cutoff: float = 5.0) -> float:
    """RMS (rad/s) of the >cutoff Hz part of joint velocity, max over joints."""
    if qd.shape[0] < 8:
        return 0.0
    F = np.fft.rfft(qd - qd.mean(0), axis=0)
    f = np.fft.rfftfreq(qd.shape[0], 1 / fs)
    F[f <= cutoff] = 0
    hp = np.fft.irfft(F, n=qd.shape[0], axis=0)
    return float(np.sqrt((hp ** 2).mean(0)).max())


def hf_ratio(qd: np.ndarray, fs: float, cutoff: float = 5.0) -> float:
    """Share of joint-velocity power above `cutoff` Hz (mean over joints that move)."""
    if qd.shape[0] < 8:
        return 0.0
    P = np.abs(np.fft.rfft(qd - qd.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(qd.shape[0], 1 / fs)
    tot = P.sum(0)
    mov = tot > 1e-6 * tot.max() if tot.max() > 0 else tot > 0
    if not mov.any():
        return 0.0
    return float(np.mean(P[f > cutoff][:, mov].sum(0) / tot[mov]))


def periodicity(speed: np.ndarray, fs: float) -> tuple[float, float]:
    """(peak autocorrelation in 0.4-2 s lag, number of cycles that fit)."""
    x = speed - speed.mean()
    if x.std() < 1e-6 or len(x) < int(1.0 * fs):
        return 0.0, 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    lo, hi = int(0.4 * fs), min(int(2.0 * fs), len(ac) - 1)
    if hi <= lo:
        return 0.0, 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    return float(ac[k]), float(len(x) / k)


# ------------------------------------------------------------------ physics proxy
class PhysicsProbe:
    """Kinematics/dynamics queries on the official A3 MJCF (root held fixed at the neutral pose)."""

    def __init__(self, mjcf_path: str, fixed_base_xml: str | None = None):
        self.m = mujoco.MjModel.from_xml_path(mjcf_path)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                              for j in ALL_JOINTS])
        self.vadr = np.array([self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                              for j in ALL_JOINTS])
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ALL_JOINTS]
        self.tau_lim = np.array([self.m.jnt_actfrcrange[i, 1] for i in jid])
        self.kp_bodies = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in KEYPOINT_BODIES]
        # fixed-base copy for forward PD tracking (root weld = drop the free joint)
        xml = open(mjcf_path).read().replace('<joint name="free_joint" type="free"/>', "")
        xml = xml.replace('<body name="pelvis_link" pos="0 0 1.3">',
                          f'<body name="pelvis_link" pos="{NEUTRAL_ROOT[0]} {NEUTRAL_ROOT[1]} {NEUTRAL_ROOT[2]}">')
        import re
        xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
        tmp = os.path.join(os.path.dirname(mjcf_path), "_a3_fixed_base.xml")
        with open(tmp, "w") as f:
            f.write(xml)
        self.mf = mujoco.MjModel.from_xml_path(tmp)
        self.df = mujoco.MjData(self.mf)
        self.f_qadr = np.array([self.mf.jnt_qposadr[mujoco.mj_name2id(self.mf, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                for j in ALL_JOINTS])
        self.f_vadr = np.array([self.mf.jnt_dofadr[mujoco.mj_name2id(self.mf, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                for j in ALL_JOINTS])
        self.f_act = np.array([mujoco.mj_name2id(self.mf, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ALL_JOINTS])
        g = np.array([a3_joint_gains(j) for j in ALL_JOINTS])
        self.kp, self.kd = g[:, 0], g[:, 1]
        self.mf.dof_armature[self.f_vadr] = g[:, 2]
        self.m.dof_armature[self.vadr] = g[:, 2]
        self._neutral_contact = None

    def _set(self, q_full: np.ndarray, qd: np.ndarray | None = None, qdd: np.ndarray | None = None):
        d = self.d
        d.qpos[:] = 0
        d.qpos[0:3] = NEUTRAL_ROOT[:3]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = q_full
        d.qvel[:] = 0
        d.qacc[:] = 0
        if qd is not None:
            d.qvel[self.vadr] = qd
        if qdd is not None:
            d.qacc[self.vadr] = qdd

    def keypoints(self, q: np.ndarray) -> np.ndarray:
        """(T, 15, 3) world positions of the stick-figure bodies."""
        out = np.zeros((q.shape[0], len(self.kp_bodies), 3))
        for t in range(q.shape[0]):
            self._set(q[t])
            mujoco.mj_kinematics(self.m, self.d)
            out[t] = self.d.xpos[self.kp_bodies]
        return out

    def self_contact(self, q_full: np.ndarray) -> dict[tuple[int, int], float]:
        self._set(q_full)
        mujoco.mj_forward(self.m, self.d)
        out: dict[tuple[int, int], float] = {}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:
                continue
            k = (min(b1, b2), max(b1, b2))
            out[k] = max(out.get(k, 0.0), -float(c.dist))
        return out

    def collision_excess(self, q: np.ndarray, stride: int = 2) -> float:
        """Max penetration (m) beyond the neutral pose's own contacts for the same body pair."""
        if self._neutral_contact is None:
            self._neutral_contact = self.self_contact(NEUTRAL_RAD)
        worst = 0.0
        for t in range(0, q.shape[0], stride):
            for k, depth in self.self_contact(q[t]).items():
                worst = max(worst, depth - self._neutral_contact.get(k, 0.0))
        return worst

    def torque_ratio(self, q: np.ndarray, fs: float) -> tuple[float, str]:
        """Fixed-root inverse dynamics: max_t,j |tau| / tau_lim over controlled joints."""
        dt = 1.0 / fs
        qd = np.gradient(q, dt, axis=0)
        qdd = np.gradient(qd, dt, axis=0)
        worst, wj = 0.0, ""
        # contacts excluded: self-contact is scored by collision_excess, and feet/floor
        # contact would otherwise leak into the arm torques
        saved = self.m.opt.disableflags
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        for t in range(q.shape[0]):
            self._set(q[t], qd[t], qdd[t])
            mujoco.mj_inverse(self.m, self.d)
            tau = self.d.qfrc_inverse[self.vadr]
            r = np.abs(tau[CTRL_IDX]) / self.tau_lim[CTRL_IDX]
            i = int(np.argmax(r))
            if r[i] > worst:
                worst, wj = float(r[i]), ALL_JOINTS[CTRL_IDX[i]]
        self.m.opt.disableflags = saved
        return worst, wj

    def pd_tracking(self, q: np.ndarray, fs: float, kp_scale: float = 1.0) -> tuple[float, float, float]:
        """Torque-limited PD + gravity/inertia feed-forward on the fixed-base model.
        Returns (rmse rad, max err rad, saturated-sample fraction) over controlled joints."""
        m, d = self.mf, self.df
        mujoco.mj_resetData(m, d)
        d.qpos[self.f_qadr] = q[0]
        mujoco.mj_forward(m, d)
        sub = max(1, int(round((1.0 / fs) / m.opt.timestep)))
        lim = self.tau_lim
        kp, kd = kp_scale * self.kp, self.kd   # SONIC A3 servo gains
        dt = 1.0 / fs
        qd_ref = np.gradient(q, dt, axis=0)
        qdd_ref = np.gradient(qd_ref, dt, axis=0)
        errs, sat, n = [], 0, 0
        d_ff = mujoco.MjData(m)
        saved = m.opt.disableflags
        for t in range(q.shape[0]):
            # feed-forward from inverse dynamics of the reference
            d_ff.qpos[:] = 0
            d_ff.qpos[self.f_qadr] = q[t]
            d_ff.qvel[self.f_vadr] = qd_ref[t]
            d_ff.qacc[self.f_vadr] = qdd_ref[t]
            m.opt.disableflags = saved | mujoco.mjtDisableBit.mjDSBL_CONTACT
            mujoco.mj_inverse(m, d_ff)
            m.opt.disableflags = saved
            ff = d_ff.qfrc_inverse[self.f_vadr]
            for _ in range(sub):
                e = q[t] - d.qpos[self.f_qadr]
                ed = qd_ref[t] - d.qvel[self.f_vadr]
                tau = ff + kp * e + kd * ed
                tau_c = np.clip(tau, -lim, lim)
                sat += int(np.any(np.abs(tau[CTRL_IDX]) > lim[CTRL_IDX]))
                n += 1
                d.ctrl[self.f_act] = tau_c
                mujoco.mj_step(m, d)
            errs.append(np.abs(q[t] - d.qpos[self.f_qadr])[CTRL_IDX])
        E = np.array(errs)
        return float(np.sqrt(np.mean(E ** 2))), float(E.max()), sat / max(n, 1)


# ------------------------------------------------------------------ per-clip evaluation
def evaluate_clip(clip_id: str, q: np.ndarray, fs: float, probe: PhysicsProbe,
                  kp: np.ndarray | None = None, physics: bool = True,
                  extra_flags: list[str] | None = None) -> tuple[ClipQuality, np.ndarray]:
    """q: (T,31) rad. Returns (quality, keypoints (T,15,3))."""
    dt = 1.0 / fs
    qc = q[:, CTRL_IDX]
    qd = np.gradient(qc, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    jerk = np.gradient(qdd, dt, axis=0)
    kp = probe.keypoints(q) if kp is None else kp
    hands = kp[:, [HAND_L, HAND_R]]
    hv = np.linalg.norm(np.gradient(hands, dt, axis=0), axis=2)          # (T,2)
    hand_path = hv.sum(0) * dt
    active = [i for i in range(2) if hand_path[i] > 0.05]
    sp = [sparc(hv[:, i], fs) for i in active] or [-1.4]
    sparc_v = float(min(sp))
    hf = hf_ratio(qd, fs)
    hfr = hf_rms(qd, fs)
    jerk_ratio = float(np.abs(jerk).max() / 2200.0)
    steps = np.abs(np.diff(qc, axis=0))
    step_max = float(steps.max()) if len(steps) else 0.0
    step_ratio = float(step_max / (np.median(steps.max(1)) + 1e-4)) if len(steps) else 0.0
    n_b = min(2, q.shape[0] - 1)
    v_in = float(np.abs(qd[:n_b + 1]).max())
    v_out = float(np.abs(qd[-n_b - 1:]).max())
    T_in = bridge_duration(NEUTRAL_RAD[CTRL_IDX], qc[0])
    T_out = bridge_duration(qc[-1], NEUTRAL_RAD[CTRL_IDX])
    lo, hi = JOINT_LIMITS_RAD[CTRL_IDX, 0], JOINT_LIMITS_RAD[CTRL_IDX, 1]
    margin = float(np.min(np.minimum(qc - lo, hi - qc)))
    extent = float(max(np.ptp(hands[:, i], axis=0).max() for i in range(2)))
    peak_speed = float(hv.max())
    per_ac, per_cycles = periodicity(hv.sum(1), fs)
    dur = q.shape[0] / fs

    met = dict(duration=round(dur, 3), sparc=round(sparc_v, 3), hf_ratio=round(hf, 4), hf_rms=round(hfr, 4),
               jerk_ratio=round(jerk_ratio, 3), step_max=round(step_max, 4), step_ratio=round(step_ratio, 2),
               v_entry=round(v_in, 3), v_exit=round(v_out, 3), bridge_in_s=round(T_in, 3),
               bridge_out_s=round(T_out, 3), limit_margin=round(margin, 4),
               hand_path_L=round(float(hand_path[0]), 3), hand_path_R=round(float(hand_path[1]), 3),
               extent=round(extent, 3), peak_hand_speed=round(peak_speed, 3),
               periodicity=round(per_ac, 3), cycles=round(per_cycles, 2))
    if physics:
        coll = probe.collision_excess(q)
        tr, tr_joint = probe.torque_ratio(q, fs)
        rmse, emax, satf = probe.pd_tracking(q, fs)
        met.update(collision_excess_mm=round(coll * 1000, 2), torque_ratio=round(tr, 3), torque_joint=tr_joint,
                   track_rmse=round(rmse, 4), track_max=round(emax, 4), sat_frac=round(satf, 4))

    flags: list[str] = list(extra_flags or [])
    # --- naturalness
    nat = 100.0
    if hfr > 0.12 and hf > 0.08: nat -= min(45, (hfr - 0.12) * 150 + 10); flags.append("jitter")
    if sparc_v < -4.0: nat -= min(35, (-4.0 - sparc_v) * 15); flags.append("unsmooth")
    if jerk_ratio > 0.8: nat -= min(25, (jerk_ratio - 0.8) * 50); flags.append("jerk")
    # --- coherence
    coh = 100.0
    if step_max > 0.15 and step_ratio > 8: coh -= 50; flags.append("spike")
    if v_in > 0.6: coh -= min(30, (v_in - 0.6) * 25); flags.append("entry_vel")
    if v_out > 0.6: coh -= min(30, (v_out - 0.6) * 25); flags.append("exit_vel")
    if T_in > 1.6: coh -= min(15, (T_in - 1.6) * 15); flags.append("far_entry")
    # --- executability
    exe = 100.0
    if margin < 0.05: exe -= 30; flags.append("limit_margin")
    if physics:
        if met["collision_excess_mm"] > 2.0: exe -= min(60, 25 + met["collision_excess_mm"] * 3); flags.append("self_collision")
        if met["torque_ratio"] > 1.0: exe -= min(45, 20 + (met["torque_ratio"] - 1) * 60); flags.append("torque")
        elif met["torque_ratio"] > 0.8: exe -= 10; flags.append("torque_near")
        if met["track_rmse"] > 0.10: exe -= min(30, (met["track_rmse"] - 0.10) * 200 + 5); flags.append("tracking")
    # --- expressiveness / gesture-likeness
    exp_ = 100.0
    if max(hand_path) < 0.12 or extent < 0.08: exp_ -= 55; flags.append("static")
    else:
        exp_ -= max(0.0, 25 - extent * 60)
    if per_ac > 0.55 and per_cycles >= 2.5: exp_ -= 30; flags.append("periodic")
    if dur < 0.5: exp_ -= 15; flags.append("too_short")
    if "style_eroded" in flags: exp_ -= 10

    sub = {k: round(float(np.clip(v, 0, 100)), 1) for k, v in
           dict(naturalness=nat, coherence=coh, executability=exe, expressiveness=exp_).items()}
    # executability is a gate: a clip the robot cannot do is worth little however nice it looks
    score = 0.30 * sub["naturalness"] + 0.25 * sub["coherence"] + 0.25 * sub["executability"] + 0.20 * sub["expressiveness"]
    # gates: a clip the robot cannot do, a broken clip, or a clip that conveys nothing is
    # worth little however good its other sub-scores look
    if sub["executability"] < 50:
        score = min(score, 54)
    if "spike" in flags:
        score = min(score, 60)
    if "static" in flags:
        score = min(score, 65)
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
    flags = sorted(set(flags), key=lambda f: -SEVERITY.get(f, 0))
    return ClipQuality(clip_id, round(float(score), 1), grade, sub, met, flags), kp


# ------------------------------------------------------------------ library-level richness
def embed(kp: np.ndarray, n: int = 24) -> np.ndarray:
    """Time-normalised hand trajectories relative to shoulders → fixed-length vector."""
    rel = np.concatenate([kp[:, HAND_L] - kp[:, SHOULDER_L], kp[:, HAND_R] - kp[:, SHOULDER_R]], 1)  # (T,6)
    t_src = np.linspace(0, 1, rel.shape[0])
    t_dst = np.linspace(0, 1, n)
    return np.stack([np.interp(t_dst, t_src, rel[:, k]) for k in range(6)], 1).ravel()


def style_descriptor(kp: np.ndarray, fs: float) -> tuple[float, float]:
    """(energy, tempo): energy = hand workspace extent (m); tempo = mean active hand speed / extent (1/s)."""
    hands = kp[:, [HAND_L, HAND_R]]
    v = np.linalg.norm(np.gradient(hands, 1 / fs, axis=0), axis=2).max(1)
    ext = max(float(np.ptp(hands[:, i], axis=0).max()) for i in range(2))
    act = v[v > 0.05]
    tempo = float(act.mean() / max(ext, 0.05)) if len(act) else 0.0
    return ext, tempo


ENERGY_BINS = [0.0, 0.12, 0.25, 0.4, 10]
TEMPO_BINS = [0.0, 1.0, 2.0, 3.5, 100]


def library_report(clips: list[dict]) -> dict:
    """clips: dicts with id, label, hand, kind, duration, quality(ClipQuality), kp, fs."""
    out: dict = {}
    labels = [c["label"] for c in clips]
    cov = {f: labels.count(f) for f in SEMANTIC_FUNCTIONS}
    present = [v for v in cov.values() if v > 0]
    p = np.array(present, float) / max(sum(present), 1)
    out["label_coverage"] = cov
    out["labels_covered"] = sum(1 for v in cov.values() if v > 0)
    out["label_balance"] = round(float(-(p * np.log(p)).sum() / np.log(len(SEMANTIC_FUNCTIONS))), 3) if len(p) else 0.0
    out["hand"] = {h: sum(1 for c in clips if c["hand"] == h) for h in ("L", "R", "B")}
    out["grades"] = {g: sum(1 for c in clips if c["quality"].grade == g) for g in "ABCD"}
    fl: dict[str, int] = {}
    for c in clips:
        for f in c["quality"].flags:
            fl[f] = fl.get(f, 0) + 1
    out["flags"] = dict(sorted(fl.items(), key=lambda kv: -kv[1]))
    out["duration_hist"] = np.histogram([c["duration"] for c in clips], bins=[0, 0.75, 1.5, 2.5, 3.5, 5, 12])[0].tolist()

    # embeddings, diversity, near-duplicates
    # distances in metres: RMS gap between time-normalised hand trajectories (relative to shoulders)
    E = np.stack([embed(c["kp"]) for c in clips])
    D = np.sqrt(((E[:, None] - E[None]) ** 2).mean(-1))
    moving = np.array(["static" not in c["quality"].flags for c in clips])
    div = {}
    for f in SEMANTIC_FUNCTIONS:
        idx = [i for i, c in enumerate(clips) if c["label"] == f]
        if len(idx) >= 2:
            sub = D[np.ix_(idx, idx)]
            div[f] = round(float(sub[np.triu_indices(len(idx), 1)].mean()), 3)   # metres
        else:
            div[f] = None
    out["intra_label_diversity"] = div
    dup_th = 0.025   # < 2.5 cm RMS apart over the whole clip, both actually moving
    dups = [(clips[i]["id"], clips[j]["id"], round(float(D[i, j]), 3))
            for i in range(len(clips)) for j in range(i + 1, len(clips))
            if D[i, j] < dup_th and moving[i] and moving[j]
            and clips[i]["kind"] == clips[j]["kind"] == "atom"]
    out["near_duplicates"] = dups[:50]
    out["near_duplicate_count"] = len(dups)

    # style space coverage (energy × tempo grid), overall and by kind
    grid = np.zeros((len(ENERGY_BINS) - 1, len(TEMPO_BINS) - 1), int)
    grid_kind = {"atom": grid.copy(), "variant": grid.copy()}
    for c in clips:
        e, t = style_descriptor(c["kp"], c["fs"])
        c["style_xy"] = (round(e, 3), round(t, 3))
        i = np.searchsorted(ENERGY_BINS, e, side="right") - 1
        j = np.searchsorted(TEMPO_BINS, t, side="right") - 1
        grid[i, j] += 1
        grid_kind["atom" if c["kind"] == "atom" else "variant"][i, j] += 1
    out["style_grid"] = grid.tolist()
    out["style_grid_by_kind"] = {k: v.tolist() for k, v in grid_kind.items()}
    out["style_cells_covered"] = int((grid > 0).sum())
    out["style_cells_total"] = int(grid.size)
    out["energy_bins"], out["tempo_bins"] = ENERGY_BINS, TEMPO_BINS

    # connectivity
    tin = np.array([c["quality"].metrics["bridge_in_s"] for c in clips])
    out["reachable_from_neutral_0p6s"] = round(float((tin <= 0.6).mean()), 3)
    out["median_bridge_in_s"] = round(float(np.median(tin)), 3)
    return out
