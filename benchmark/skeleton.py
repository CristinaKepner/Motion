"""Joint-name profiles for public gesture datasets, plus fuzzy auto-matching.

The retargeter needs nine landmarks: hips, chest (upper spine), neck, and per side
shoulder (glenohumeral joint), elbow, wrist, hand end (a finger base or hand End Site).
Profiles list the usual names; `resolve()` falls back to pattern matching so an unknown
BVH still maps as long as its names contain the usual words.
"""
from __future__ import annotations

import re

LANDMARKS = ["hips", "chest", "neck", "l_shoulder", "l_elbow", "l_wrist", "l_hand_end",
             "r_shoulder", "r_elbow", "r_wrist", "r_hand_end"]

PROFILES: dict[str, dict] = {
    # BEAT v1 BVH (Vicon Shogun export) / Trinity / ZEGGS share the Mixamo-style naming
    "beat": {"hips": "Hips", "chest": "Spine3", "neck": "Neck", "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm",
             "l_wrist": "LeftHand", "l_hand_end": "LeftHandMiddle1", "r_shoulder": "RightArm", "r_elbow": "RightForeArm",
             "r_wrist": "RightHand", "r_hand_end": "RightHandMiddle1", "unit_scale": 0.01, "fps": 120},
    "zeggs": {"hips": "Hips", "chest": "Spine3", "neck": "Neck", "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm",
              "l_wrist": "LeftHand", "l_hand_end": "LeftHandMiddle1", "r_shoulder": "RightArm", "r_elbow": "RightForeArm",
              "r_wrist": "RightHand", "r_hand_end": "RightHandMiddle1", "unit_scale": 0.01, "fps": 60},
    "trinity": {"hips": "Hips", "chest": "Spine3", "neck": "Neck", "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm",
                "l_wrist": "LeftHand", "l_hand_end": "LeftHand_End", "r_shoulder": "RightArm", "r_elbow": "RightForeArm",
                "r_wrist": "RightHand", "r_hand_end": "RightHand_End", "unit_scale": 0.01, "fps": 60},
    # Talking With Hands 16.2M / GENEA 2022-2023 ("b_" prefix)
    "twh": {"hips": "body_world", "chest": "b_spine3", "neck": "b_neck0", "l_shoulder": "b_l_arm", "l_elbow": "b_l_forearm",
            "l_wrist": "b_l_wrist", "l_hand_end": "b_l_middle1", "r_shoulder": "b_r_arm", "r_elbow": "b_r_forearm",
            "r_wrist": "b_r_wrist", "r_hand_end": "b_r_middle1", "unit_scale": 0.01, "fps": 30},
    # Bandai Namco Research Motiondataset 1/2 (Y-up, cm, 30 Hz; no fingers → hand End Site)
    "bandai": {"hips": "Hips", "chest": "Chest", "neck": "Neck", "l_shoulder": "UpperArm_L", "l_elbow": "LowerArm_L",
               "l_wrist": "Hand_L", "l_hand_end": "Hand_L_End", "r_shoulder": "UpperArm_R", "r_elbow": "LowerArm_R",
               "r_wrist": "Hand_R", "r_hand_end": "Hand_R_End", "unit_scale": 0.01, "fps": 30},
    # HiPHI 55-joint standard BVH (Y-up, cm, 90 Hz)
    "hiphi": {"hips": "Hips", "chest": "Chest", "neck": "Neck", "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm",
              "l_wrist": "LeftHand", "l_hand_end": "LeftHandMiddle1", "r_shoulder": "RightArm", "r_elbow": "RightForeArm",
              "r_wrist": "RightHand", "r_hand_end": "RightHandMiddle1", "unit_scale": 0.01, "fps": 90},
}

_PATTERNS = {
    "hips": [r"^hips?$", r"pelvis", r"root", r"body_world"],
    "chest": [r"spine3$", r"^chest$", r"spine2$", r"upperchest", r"spine1$", r"^spine$"],
    "neck": [r"neck0?$", r"neck1$"],
    "l_shoulder": [r"^left_?arm$", r"l_?arm$", r"left.*upperarm", r"l_?upperarm", r"left_?shoulder$", r"lshoulder$"],
    "l_elbow": [r"left_?forearm", r"l_?forearm", r"left.*elbow", r"l_?elbow", r"leftlowerarm"],
    "l_wrist": [r"^left_?hand$", r"l_?wrist$", r"^l_?hand$", r"left.*wrist"],
    "l_hand_end": [r"left_?handmiddle1", r"l_?middle1", r"left_?hand_end", r"l_?hand_end", r"left_?handindex1", r"l_?index1"],
    "r_shoulder": [r"^right_?arm$", r"r_?arm$", r"right.*upperarm", r"r_?upperarm", r"right_?shoulder$", r"rshoulder$"],
    "r_elbow": [r"right_?forearm", r"r_?forearm", r"right.*elbow", r"r_?elbow", r"rightlowerarm"],
    "r_wrist": [r"^right_?hand$", r"r_?wrist$", r"^r_?hand$", r"right.*wrist"],
    "r_hand_end": [r"right_?handmiddle1", r"r_?middle1", r"right_?hand_end", r"r_?hand_end", r"right_?handindex1", r"r_?index1"],
}


def resolve(names: list[str], profile: str | dict | None = None) -> dict[str, str]:
    """Map landmarks → joint names present in `names`. Profile names tried first, then patterns."""
    prof = PROFILES.get(profile, {}) if isinstance(profile, str) else (profile or {})
    out: dict[str, str] = {}
    lower = {n.lower(): n for n in names}
    for lm in LANDMARKS:
        if lm in prof and prof[lm] in names:
            out[lm] = prof[lm]
            continue
        for pat in _PATTERNS[lm]:
            hit = [lower[n] for n in lower if re.search(pat, n)]
            if hit:
                out[lm] = sorted(hit, key=len)[0]
                break
    missing = [lm for lm in LANDMARKS if lm not in out]
    if missing:
        raise KeyError(f"could not resolve landmarks {missing} in skeleton {names[:12]}...")
    return out
