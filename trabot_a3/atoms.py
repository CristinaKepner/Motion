"""Offline Motion-Atom construction (TRABot §III-B, Eq. 4-10).

Pipeline per recording
  1. activity energy E_t over the 14 arm joints                        (Eq. 4)
  2. per-recording background scale b, Δ, σ; reject low-signal clips  (Eq. 5-7)
  3. hysteresis on/off thresholds + duration confirmation + gap merge  (Eq. 8)
  4. boundary refinement to slow/stable/low-posture-change cut points  (Eq. 9)
     long spans split with a duration-penalised DP into 0.4-4 s atoms
  5. kinematic validation on the 30 Hz execution copy; repair by uniform
     time stretch s and amplitude scaling; position-limit margin        (Eq. 10)
  6. semantic label proposal (rule-based stand-in for Qwen3-VL) + handedness,
     flagged for the two human-review passes the paper requires.

Detection copy: 25 Hz, 3-frame median then [1,4,6,4,1]/16.
Execution copy: 30 Hz, untouched by the detection smoothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .data import (A3Motion, ARM_IDX, LEFT_ARM_IDX, RIGHT_ARM_IDX, JOINT_LIMITS_RAD,
                   NEUTRAL_RAD, ALL_JOINTS, CTRL_IDX, load_a3_csv)

# Joints the gesture layer actually commands: waist (3) + both arms (14).
# Head and legs are left to the neutral pose / the whole-body controller, so
# their limits are not part of atom validation (the recordings hit ankle
# limits during gait, which is irrelevant to an upper-body gesture asset).
CONTROLLED_IDX = CTRL_IDX

SEMANTIC_FUNCTIONS = [
    "emphasize", "enumerate", "contrast_compare",
    "refer_self", "refer_other", "refer_external",
    "depict_size_shape", "depict_direction_motion",
    "turn_open", "turn_yield",
    "greet_farewell", "thank", "affirm", "deny",
    "certainty", "uncertainty", "enthusiasm",
]


@dataclass
class CatalogParams:
    # Eq.4-8 detection
    detect_fps: float = 25.0
    exec_fps: float = 30.0
    kappa_on: float = 3.0
    kappa_off: float = 1.5
    alpha_on: float = 0.20
    alpha_off: float = 0.08
    beta_on: float = 0.55
    beta_off: float = 0.25
    confirm_on_s: float = 0.12
    confirm_off_s: float = 0.16
    merge_gap_s: float = 0.12
    hold_recover_s: float = 1.0
    hold_posture_rad: float = 0.15      # mean |q-q_neutral| over arm joints to count as "held away from neutral"
    min_signal_sigma_mult: float = 4.0  # reject recording if Δ < 4σ
    # Eq.9 boundary refinement
    lam_E: float = 0.60
    lam_A: float = 0.20
    lam_P: float = 0.20
    search_window_s: float = 0.32
    max_boundary_cost: float = 0.50
    min_atom_s: float = 0.4
    max_atom_s: float = 4.0
    target_atom_s: float = 2.0
    posture_window_s: float = 0.10
    # Eq.10 validation / repair
    v_lim: float = 7.0       # rad/s
    a_lim: float = 120.0     # rad/s^2
    j_lim: float = 2200.0    # rad/s^3
    stretch_margin: float = 1.02
    max_stretch: float = 3.0
    max_repaired_s: float = 10.0
    pos_margin_rad: float = 0.05


@dataclass
class MotionAtom:
    """a_i = (φ_i, Q_i, d_i, h_i) plus provenance (Eq. 1-3)."""
    atom_id: str
    label: str                      # φ_i  (proposed; human review pending unless reviewed=True)
    hand: str                       # h_i  in {"L","R","B"}
    duration: float                 # d_i  seconds, after repair
    fps: float
    q: np.ndarray                   # (T,31) rad execution trajectory, 30 Hz
    root: np.ndarray                # (T,6)
    source: str
    src_start_s: float
    src_end_s: float
    stretch: float = 1.0
    amp_scale: float = 1.0
    boundary_cost: tuple[float, float] = (0.0, 0.0)
    reviewed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def q_in(self) -> np.ndarray:     # entry pose, controlled joints (waist + arms)
        return self.q[0, CTRL_IDX]

    @property
    def q_out(self) -> np.ndarray:    # exit pose, controlled joints
        return self.q[-1, CTRL_IDX]

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("q"); d.pop("root")
        d["q_in"] = self.q_in.round(4).tolist()
        d["q_out"] = self.q_out.round(4).tolist()
        d["frames"] = int(self.q.shape[0])
        return d


# --------------------------------------------------------------------------- Eq. 4
def activity_energy(q_arm: np.ndarray, fps: float, ranges: np.ndarray) -> np.ndarray:
    """E_t = sqrt( 1/14 Σ_j ( qdot_{t,j} / r_j )^2 ),  r_j = joint range (rad)."""
    qd = np.gradient(q_arm, 1.0 / fps, axis=0)
    return np.sqrt(np.mean((qd / ranges) ** 2, axis=1))


def detection_copy(E: np.ndarray) -> np.ndarray:
    """3-frame median, then binomial [1,4,6,4,1]/16 smoothing (paper's detection copy)."""
    Em = median_filter(E, size=3, mode="nearest")
    k = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
    return np.convolve(np.pad(Em, 2, mode="edge"), k, mode="valid")


# --------------------------------------------------------------------------- Eq. 5-8
@dataclass
class BackgroundStats:
    b: float
    delta: float
    sigma: float
    theta_on: float
    theta_off: float
    accepted: bool


def background_stats(E: np.ndarray, p: CatalogParams) -> BackgroundStats:
    b = float(np.percentile(E, 20))                       # Eq.5
    delta = float(np.percentile(E, 95) - b)
    quiet = E[E <= np.percentile(E, 50)]                  # Eq.6
    med = np.median(quiet)
    sigma = float(1.4826 * np.median(np.abs(quiet - med)))  # Eq.7 (MAD)
    accepted = delta >= p.min_signal_sigma_mult * sigma

    def theta(kappa, alpha, beta):                        # Eq.8
        return b + float(np.clip(max(kappa * sigma, alpha * delta), alpha * delta, beta * delta))

    return BackgroundStats(b, delta, sigma,
                           theta(p.kappa_on, p.alpha_on, p.beta_on),
                           theta(p.kappa_off, p.alpha_off, p.beta_off), accepted)


def hysteresis_segments(E: np.ndarray, posture_dev: np.ndarray, st: BackgroundStats,
                        fps: float, p: CatalogParams) -> list[tuple[int, int]]:
    """State machine with on/off duration confirmation, gap merge and hold preservation.

    posture_dev: per-frame mean |q_arm - q_neutral| (rad) — used for the hold rule.
    """
    n_on = max(1, int(round(p.confirm_on_s * fps)))
    n_off = max(1, int(round(p.confirm_off_s * fps)))
    segs: list[tuple[int, int]] = []
    active = False
    start = 0
    cnt = 0
    for t, e in enumerate(E):
        if not active:
            if e > st.theta_on:
                cnt += 1
                if cnt >= n_on:
                    active = True
                    start = t - n_on + 1
                    cnt = 0
            else:
                cnt = 0
        else:
            if e < st.theta_off:
                cnt += 1
                if cnt >= n_off:
                    active = False
                    segs.append((start, t - n_off + 1))
                    cnt = 0
            else:
                cnt = 0
    if active:
        segs.append((start, len(E)))

    # merge inactive gaps shorter than merge_gap_s; also merge "hold" gaps:
    # posture held away from neutral and motion resumes within hold_recover_s.
    merged: list[tuple[int, int]] = []
    gap_f = int(round(p.merge_gap_s * fps))
    hold_f = int(round(p.hold_recover_s * fps))
    for s, e in segs:
        if merged:
            ps, pe = merged[-1]
            gap = s - pe
            held = gap <= hold_f and np.mean(posture_dev[pe:s]) > p.hold_posture_rad
            if gap < gap_f or held:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    return merged


# --------------------------------------------------------------------------- Eq. 9
def boundary_cost(E: np.ndarray, q_arm: np.ndarray, fps: float, st: BackgroundStats,
                  p: CatalogParams) -> np.ndarray:
    """C^bdry_t = λ_E Ē_t + λ_A Ā_t + λ_P P̄_t, all three normalised to ~[0,1]."""
    E_bar = np.clip((E - st.b) / max(st.delta, 1e-9), 0.0, 1.5)
    A = np.abs(np.gradient(E, 1.0 / fps))
    A_bar = np.clip(A / max(np.percentile(A, 95), 1e-9), 0.0, 1.5)
    w = max(1, int(round(p.posture_window_s * fps)))
    qp = np.pad(q_arm, ((w, w), (0, 0)), mode="edge")
    ranges = JOINT_LIMITS_RAD[ARM_IDX, 1] - JOINT_LIMITS_RAD[ARM_IDX, 0]
    P = np.mean(np.abs(qp[2 * w:] - qp[:-2 * w]) / ranges, axis=1)
    P_bar = np.clip(P / max(np.percentile(P, 95), 1e-9), 0.0, 1.5)
    return p.lam_E * E_bar + p.lam_A * A_bar + p.lam_P * P_bar


def refine_boundaries(seg: tuple[int, int], C: np.ndarray, fps: float, p: CatalogParams
                      ) -> tuple[tuple[int, int], tuple[float, float]] | None:
    """Move start backward / end forward (≤ search_window_s) to the lowest-cost frame."""
    s, e = seg
    w = int(round(p.search_window_s * fps))
    s0 = max(0, s - w)
    s_new = s0 + int(np.argmin(C[s0:s + 1]))
    e1 = min(len(C), e + w + 1)
    e_new = e + int(np.argmin(C[e:e1]))
    cs, ce = float(C[s_new]), float(C[min(e_new, len(C) - 1)])
    if cs > p.max_boundary_cost or ce > p.max_boundary_cost:
        return None
    return (s_new, e_new), (cs, ce)


def split_long_segment(seg: tuple[int, int], C: np.ndarray, fps: float, p: CatalogParams
                       ) -> list[tuple[int, int]]:
    """Duration-penalised DP over candidate cut frames (local cost minima)."""
    s, e = seg
    L = e - s
    min_f, max_f, tgt_f = int(p.min_atom_s * fps), int(p.max_atom_s * fps), p.target_atom_s * fps
    if L <= max_f:
        return [seg]
    # candidate cuts: local minima of C inside the span with cost below threshold
    cand = [t for t in range(s + min_f, e - min_f)
            if C[t] <= p.max_boundary_cost and C[t] <= C[t - 1] and C[t] <= C[t + 1]]
    nodes = [s] + cand + [e]
    INF = float("inf")
    best = [INF] * len(nodes)
    prev = [-1] * len(nodes)
    best[0] = 0.0
    for j in range(1, len(nodes)):
        for i in range(j):
            d = nodes[j] - nodes[i]
            if d < min_f or d > max_f:
                continue
            cost = best[i] + (C[nodes[j]] if j < len(nodes) - 1 else 0.0) + 0.3 * abs(d - tgt_f) / tgt_f
            if cost < best[j]:
                best[j], prev[j] = cost, i
    if best[-1] == INF:
        # fall back: uniform split
        k = int(np.ceil(L / max_f))
        cuts = np.linspace(s, e, k + 1).astype(int)
        return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
    out, j = [], len(nodes) - 1
    while j > 0:
        out.append((nodes[prev[j]], nodes[j]))
        j = prev[j]
    return out[::-1]


# --------------------------------------------------------------------------- Eq. 10
@dataclass
class KinematicReport:
    v_max: float
    a_max: float
    j_max: float
    stretch: float
    limit_violation: float   # max rad beyond (limit ∓ margin), 0 if clean


def kinematic_report(q: np.ndarray, fps: float, p: CatalogParams) -> KinematicReport:
    dt = 1.0 / fps
    v = np.gradient(q, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    qc, vc, ac, jc = (x[:, CONTROLLED_IDX] for x in (q, v, a, j))
    vm, am, jm = np.abs(vc).max(), np.abs(ac).max(), np.abs(jc).max()
    s_inner = max(1.0, vm / p.v_lim, np.sqrt(am / p.a_lim), np.cbrt(jm / p.j_lim))
    # Eq.10 literally reads s = 1.02·max{1, ...}; we apply the 2 % margin only when a
    # repair is actually needed, so clean atoms keep their recorded timing.
    s = p.stretch_margin * s_inner if s_inner > 1.0 + 1e-9 else 1.0
    lo = JOINT_LIMITS_RAD[CONTROLLED_IDX, 0] + p.pos_margin_rad
    hi = JOINT_LIMITS_RAD[CONTROLLED_IDX, 1] - p.pos_margin_rad
    viol = float(max(0.0, (lo - qc).max(), (qc - hi).max()))
    return KinematicReport(float(vm), float(am), float(jm), float(s), viol)


def time_stretch(m: A3Motion, s: float) -> A3Motion:
    """Uniform stretch T' = sT: v' = v/s, a' = a/s², j' = j/s³ (paper's derivative relation)."""
    if s <= 1.0 + 1e-9:
        return m
    n = int(round(m.T * s))
    t_src = np.linspace(0, 1, m.T)
    t_dst = np.linspace(0, 1, n)
    q = np.stack([np.interp(t_dst, t_src, m.q[:, k]) for k in range(m.q.shape[1])], 1)
    root = np.stack([np.interp(t_dst, t_src, m.root[:, k]) for k in range(6)], 1)
    return A3Motion(m.name, m.fps, root, q)


def amplitude_scale(q: np.ndarray, scale: float) -> np.ndarray:
    """Shrink arm motion about the neutral pose (paper: 缩幅)."""
    out = q.copy()
    out[:, CONTROLLED_IDX] = NEUTRAL_RAD[CONTROLLED_IDX] + scale * (q[:, CONTROLLED_IDX] - NEUTRAL_RAD[CONTROLLED_IDX])
    return out


# --------------------------------------------------------------------------- labels
def handedness(q_arm: np.ndarray, fps: float) -> str:
    ranges = JOINT_LIMITS_RAD[ARM_IDX, 1] - JOINT_LIMITS_RAD[ARM_IDX, 0]
    qd = np.abs(np.gradient(q_arm, 1.0 / fps, axis=0)) / ranges
    l, r = qd[:, :7].sum(), qd[:, 7:].sum()
    share = l / max(l + r, 1e-9)
    return "L" if share > 0.7 else "R" if share < 0.3 else "B"


def propose_label(source: str, hand: str, q_arm: np.ndarray) -> tuple[str, str]:
    """Rule-based stand-in for the Qwen3-VL label proposal. Returns (label, rationale).

    The paper has a VLM propose one of 17 functions from the rendered clip; here we
    only have file names + kinematics, so this is deliberately coarse and every atom
    is marked reviewed=False.
    """
    s = source.lower()
    amp = float(np.mean(np.ptp(q_arm, axis=0)))
    raise_pitch = float(-(q_arm[:, [0, 7]].min()))   # shoulder pitch negative = arm forward/up
    if "overhead" in s:
        return "emphasize", "both arms raised overhead → strong emphasis beat"
    if "reach_front" in s:
        return "depict_size_shape", "both hands extended forward → shape/size depiction"
    if "punch" in s:
        return "emphasize", "alternating forward thrusts → emphatic beats"
    if "bend_side" in s or "twist" in s:
        return "depict_direction_motion", "torso-led lateral motion → direction depiction"
    if "shift" in s:
        return "uncertainty", "weight shift / sway → hedging"
    if "rotate" in s:
        return "refer_external", "turning in place → referring to surroundings"
    if "balance" in s:
        return "certainty", "single-leg hold, arms steady → stance"
    if "squat" in s or "lunge" in s:
        return "enumerate", "repeated counted reps → enumerate beats"
    if "walk" in s or "march" in s or "step" in s:
        return "turn_open" if hand == "B" else "turn_yield", "arm swing with gait → turn-taking beat"
    return ("enthusiasm" if amp > 0.5 else "affirm"), f"amplitude {amp:.2f}"


# --------------------------------------------------------------------------- catalog
@dataclass
class AtomCatalog:
    atoms: list[MotionAtom]
    rejected: list[dict]
    params: CatalogParams
    recordings: dict[str, dict]

    def by_label(self, label: str) -> list[MotionAtom]:
        return [a for a in self.atoms if a.label == label]

    def labels(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.atoms:
            out[a.label] = out.get(a.label, 0) + 1
        return dict(sorted(out.items()))

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {"params": asdict(self.params), "recordings": self.recordings,
                "atoms": [a.meta() for a in self.atoms], "rejected": self.rejected}
        (out / "catalog.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        for a in self.atoms:
            np.savez_compressed(out / f"{a.atom_id}.npz", q=a.q, root=a.root)


def build_catalog(csv_paths: list[str | Path], p: CatalogParams | None = None,
                  verbose: bool = True) -> AtomCatalog:
    p = p or CatalogParams()
    ranges = JOINT_LIMITS_RAD[ARM_IDX, 1] - JOINT_LIMITS_RAD[ARM_IDX, 0]
    atoms: list[MotionAtom] = []
    rejected: list[dict] = []
    recs: dict[str, dict] = {}

    for path in csv_paths:
        src = load_a3_csv(path)
        det = src.resample(p.detect_fps)
        exe = src.resample(p.exec_fps)
        E_raw = activity_energy(det.q[:, ARM_IDX], det.fps, ranges)
        E = detection_copy(E_raw)
        st = background_stats(E, p)
        rec = {"duration_s": round(src.duration, 2), "b": round(st.b, 4), "delta": round(st.delta, 4),
               "sigma": round(st.sigma, 4), "theta_on": round(st.theta_on, 4),
               "theta_off": round(st.theta_off, 4), "accepted": st.accepted, "atoms": 0, "rejected": 0}
        recs[src.name] = rec
        if not st.accepted:
            rejected.append({"source": src.name, "reason": f"low signal: Δ={st.delta:.3f} < 4σ={4*st.sigma:.3f}"})
            if verbose:
                print(f"[{src.name}] rejected (Δ<4σ)")
            continue
        posture_dev = np.mean(np.abs(det.q[:, ARM_IDX] - NEUTRAL_RAD[ARM_IDX]), axis=1)
        coarse = hysteresis_segments(E, posture_dev, st, det.fps, p)
        C = boundary_cost(E, det.q[:, ARM_IDX], det.fps, st, p)

        n_atoms = 0
        for seg in coarse:
            r = refine_boundaries(seg, C, det.fps, p)
            if r is None:
                rejected.append({"source": src.name, "reason": "boundary cost > 0.50",
                                 "span_s": [seg[0] / det.fps, seg[1] / det.fps]})
                rec["rejected"] += 1
                continue
            (s_d, e_d), (cs, ce) = r
            pieces = split_long_segment((s_d, e_d), C, det.fps, p)
            for (a_d, b_d) in pieces:
                dur = (b_d - a_d) / det.fps
                if dur < p.min_atom_s:
                    rejected.append({"source": src.name, "reason": f"too short {dur:.2f}s",
                                     "span_s": [a_d / det.fps, b_d / det.fps]})
                    rec["rejected"] += 1
                    continue
                # map detection timestamps -> execution frames (never frame index reuse)
                t0, t1 = a_d / det.fps, b_d / det.fps
                a_e, b_e = int(round(t0 * exe.fps)), int(round(t1 * exe.fps))
                clip = exe.slice(a_e, b_e)
                clip = A3Motion(clip.name, clip.fps, clip.root - clip.root[0:1] * [1, 1, 0, 0, 0, 0], clip.q)
                notes = []
                # Eq.10 validation + repair loop
                amp = 1.0
                rep = kinematic_report(clip.q, clip.fps, p)
                for _ in range(3):
                    if rep.limit_violation <= 0:
                        break
                    amp *= 0.85
                    clip = A3Motion(clip.name, clip.fps, clip.root, amplitude_scale(exe.slice(a_e, b_e).q, amp))
                    rep = kinematic_report(clip.q, clip.fps, p)
                    notes.append(f"amplitude scaled to {amp:.2f} for position-limit margin")
                if rep.limit_violation > 0:
                    rejected.append({"source": src.name, "reason": f"position limit violated by {rep.limit_violation:.3f} rad",
                                     "span_s": [t0, t1]})
                    rec["rejected"] += 1
                    continue
                s = rep.stretch
                if s > 1.0 + 1e-6:
                    if s > p.max_stretch or dur * s > p.max_repaired_s:
                        rejected.append({"source": src.name, "reason": f"stretch {s:.2f} too large", "span_s": [t0, t1]})
                        rec["rejected"] += 1
                        continue
                    clip = time_stretch(clip, s)
                    notes.append(f"time-stretched ×{s:.3f} (v {rep.v_max:.1f}, a {rep.a_max:.0f}, j {rep.j_max:.0f})")
                    rep2 = kinematic_report(clip.q, clip.fps, p)
                    if rep2.stretch > 1.0 + 1e-3:   # discretisation residue: re-check
                        notes.append(f"residual stretch {rep2.stretch:.3f} after resample")
                hand = handedness(clip.q[:, ARM_IDX], clip.fps)
                label, why = propose_label(src.name, hand, clip.q[:, ARM_IDX])
                atom = MotionAtom(
                    atom_id=f"{src.name}__{a_d/det.fps:05.2f}-{b_d/det.fps:05.2f}",
                    label=label, hand=hand, duration=clip.T / clip.fps, fps=clip.fps,
                    q=clip.q, root=clip.root, source=src.name, src_start_s=t0, src_end_s=t1,
                    stretch=max(1.0, s), amp_scale=amp, boundary_cost=(round(cs, 3), round(ce, 3)),
                    reviewed=False, notes=[f"label proposal: {why}"] + notes)
                atoms.append(atom)
                n_atoms += 1
        rec["atoms"] = n_atoms
        if verbose:
            print(f"[{src.name}] b={st.b:.3f} Δ={st.delta:.3f} σ={st.sigma:.3f} "
                  f"θon={st.theta_on:.3f} θoff={st.theta_off:.3f} coarse={len(coarse)} atoms={n_atoms}")
    return AtomCatalog(atoms, rejected, p, recs)


def load_catalog(dir_: str | Path) -> AtomCatalog:
    d = Path(dir_)
    meta = json.loads((d / "catalog.json").read_text())
    p = CatalogParams(**meta["params"])
    atoms = []
    for m in meta["atoms"]:
        z = np.load(d / f"{m['atom_id']}.npz")
        atoms.append(MotionAtom(
            atom_id=m["atom_id"], label=m["label"], hand=m["hand"], duration=m["duration"], fps=m["fps"],
            q=z["q"], root=z["root"], source=m["source"], src_start_s=m["src_start_s"], src_end_s=m["src_end_s"],
            stretch=m["stretch"], amp_scale=m["amp_scale"], boundary_cost=tuple(m["boundary_cost"]),
            reviewed=m["reviewed"], notes=m["notes"]))
    return AtomCatalog(atoms, meta["rejected"], p, meta["recordings"])
