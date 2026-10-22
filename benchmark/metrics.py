"""Benchmark metrics for speech-gesture data and generators on A3.

All motion metrics take A3 joint angles (T,31) rad at 30 Hz, so a dataset, a generator and the
robot's own library are scored on the same footing. Keypoints come from the A3 forward
kinematics (quality.PhysicsProbe).

Distribution   FKD   Fréchet distance between Gaussians of handcrafted kinematic features
                     (the same construction as FGD; a learned embedding can be plugged in)
               L1Div mean L1 distance of hand trajectories to the set mean (EMAGE "L1div")
Rhythm         BC    Beat Consistency: audio onsets vs motion beats (velocity minima), σ = 0.3 s
               BAS   Binary / Gaussian beat-apex scores (TRABot §IV-A, windows ±0.2 s, σ = 0.3 s)
Semantics      spec pass rate per communicative function (trabot_a3.semantics), verified refs
Executability  grade mix, torque-over-limit share, self-collision share, static share (quality.py)
Coverage       joint-range share, style-grid cells, label coverage
Adaptation     retarget fidelity (benchmark.retarget)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import sqrtm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from trabot_a3.data import CTRL_IDX, ARM_IDX, NEUTRAL_RAD, JOINT_LIMITS_RAD
from trabot_a3.quality import HAND_L, HAND_R, SHOULDER_L, SHOULDER_R, style_descriptor, ENERGY_BINS, TEMPO_BINS
from trabot_a3.semantics import check_all, SPECS


# ------------------------------------------------------------------ features / FKD
def kin_features(q: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Clip → feature vector (74-d): joint-angle mean/std, speed mean/std, hand stats."""
    qc = q[:, CTRL_IDX]
    qd = np.abs(np.gradient(qc, 1 / fs, axis=0))
    return np.concatenate([qc.mean(0), qc.std(0), qd.mean(0), qd.std(0),
                           [qd.max(), q.shape[0] / fs]])


def frechet(A: np.ndarray, B: np.ndarray) -> float:
    """Fréchet distance between the Gaussians fitted to feature sets A (n,d), B (m,d)."""
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False) + 1e-6 * np.eye(A.shape[1]), np.cov(B, rowvar=False) + 1e-6 * np.eye(B.shape[1])
    covmean = sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def fkd(qs_a: list[np.ndarray], qs_b: list[np.ndarray], fs: float = 30.0) -> float:
    return frechet(np.stack([kin_features(q, fs) for q in qs_a]), np.stack([kin_features(q, fs) for q in qs_b]))


# ------------------------------------------------------------------ diversity
def hand_traj(kp: np.ndarray, n: int = 24) -> np.ndarray:
    rel = np.concatenate([kp[:, HAND_L] - kp[:, SHOULDER_L], kp[:, HAND_R] - kp[:, SHOULDER_R]], 1)
    t_src, t_dst = np.linspace(0, 1, rel.shape[0]), np.linspace(0, 1, n)
    return np.stack([np.interp(t_dst, t_src, rel[:, k]) for k in range(6)], 1).ravel()


def l1_diversity(kps: list[np.ndarray]) -> float:
    X = np.stack([hand_traj(k) for k in kps])
    return float(np.abs(X - X.mean(0)).mean())


# ------------------------------------------------------------------ rhythm
def audio_onsets(wav_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(onset times s, onset strength peaks times s) via librosa."""
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time", backtrack=False)
    # stress proxy: strongest 30 % of onsets
    strengths = env[librosa.time_to_frames(on, sr=sr).clip(0, len(env) - 1)] if len(on) else np.array([])
    thr = np.percentile(strengths, 70) if len(strengths) else 0
    return on, on[strengths >= thr] if len(on) else on


def transcript_onsets(tsv_path: str) -> np.ndarray:
    """Word start times from a GENEA/TWH style TSV (start\tend\tword)."""
    t = []
    with open(tsv_path) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2:
                try:
                    t.append(float(p[0]))
                except ValueError:
                    pass
    return np.array(t)


def motion_beats(q: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """EMAGE/BEAT-style: local minima of summed joint angular speed."""
    v = np.abs(np.gradient(q[:, CTRL_IDX], 1 / fs, axis=0)).sum(1)
    v = gaussian_filter1d(v, 1.0)
    idx, _ = find_peaks(-v, distance=max(1, int(0.15 * fs)))
    return idx / fs


def hand_apexes(kp: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """TRABot apex: peaks of hand speed (max of both hands)."""
    hv = np.linalg.norm(np.gradient(kp[:, [HAND_L, HAND_R]], 1 / fs, axis=0), axis=2).max(1)
    hv = gaussian_filter1d(hv, 1.0)
    idx, _ = find_peaks(hv, height=max(0.15, 0.3 * hv.max()), distance=max(1, int(0.2 * fs)))
    return idx / fs


def beat_consistency(audio_beats: np.ndarray, m_beats: np.ndarray, sigma: float = 0.3) -> float:
    if len(audio_beats) == 0 or len(m_beats) == 0:
        return float("nan")
    d = np.abs(audio_beats[:, None] - m_beats[None, :]).min(1)
    return float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())


def bas(stress: np.ndarray, apex: np.ndarray, window: float = 0.2, sigma: float = 0.3) -> tuple[float, float]:
    """(Binary BAS, Gaussian BAS) — TRABot §IV-A as written up in the report."""
    if len(stress) == 0 or len(apex) == 0:
        return float("nan"), float("nan")
    d = np.abs(stress[:, None] - apex[None, :]).min(1)
    return float((d <= window).mean()), float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())


# ------------------------------------------------------------------ semantics / coverage
def semantic_summary(kps: list[np.ndarray], qs: list[np.ndarray], labels: list[str | None], fs: float = 30.0) -> dict:
    pass_fn: dict[str, int] = {fn: 0 for fn in SPECS}
    own_pass, own_n = 0, 0
    for kp, q, lab in zip(kps, qs, labels):
        R = check_all(kp, q, fs)
        for fn, r in R.items():
            pass_fn[fn] += int(r.passed)
        if lab in R:
            own_n += 1
            own_pass += int(R[lab].passed)
    n = max(len(kps), 1)
    return {"clips": len(kps), "verified_refs_per_function": pass_fn,
            "functions_with_refs": sum(1 for v in pass_fn.values() if v > 0),
            "any_function_rate": round(sum(1 for kp, q in zip(kps, qs)
                                           if any(r.passed for r in check_all(kp, q, fs).values())) / n, 3),
            "own_label_pass": (own_pass, own_n)}


def joint_coverage(qs: list[np.ndarray]) -> dict:
    Q = np.concatenate([q[:, CTRL_IDX] for q in qs], 0)
    lo, hi = JOINT_LIMITS_RAD[CTRL_IDX, 0], JOINT_LIMITS_RAD[CTRL_IDX, 1]
    share = (np.minimum(Q.max(0), hi) - np.maximum(Q.min(0), lo)) / (hi - lo)
    return {"mean_share": round(float(share.mean()), 3), "min_share": round(float(share.min()), 3),
            "per_joint": [round(float(s), 3) for s in share]}


def style_cells(kps: list[np.ndarray], fs: float = 30.0) -> dict:
    grid = np.zeros((len(ENERGY_BINS) - 1, len(TEMPO_BINS) - 1), int)
    for kp in kps:
        e, t = style_descriptor(kp, fs)
        grid[np.searchsorted(ENERGY_BINS, e, side="right") - 1, np.searchsorted(TEMPO_BINS, t, side="right") - 1] += 1
    return {"cells_covered": int((grid > 0).sum()), "cells_total": int(grid.size), "grid": grid.tolist()}
