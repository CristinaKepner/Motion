"""Transition bridges between the measured pose and an atom's entry pose (Eq. 13-14).

q_b(t) = q0 + s(t/T_b)(q1 - q0),  s(u) = 10u^3 - 15u^4 + 6u^5
T_b >= max{ 1.875 D / v_max,  sqrt((10/sqrt(3)) D / a_max),  cbrt(60 D / j_max) },  D = max_j |q1_j - q0_j|

The paper fixes v_max = 1 rad/s for online bridging (Eq. 14). The acceleration
and jerk terms are the report's extension and are off by default (a_max/j_max=None).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BridgeLimits:
    v_max: float = 1.0
    a_max: float | None = None
    j_max: float | None = None
    min_duration: float = 0.0


def quintic_s(u: np.ndarray) -> np.ndarray:
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def bridge_duration(q0: np.ndarray, q1: np.ndarray, lim: BridgeLimits = BridgeLimits()) -> float:
    D = float(np.max(np.abs(np.asarray(q1) - np.asarray(q0))))
    if D < 1e-9:
        return lim.min_duration
    T = 1.875 * D / lim.v_max
    if lim.a_max:
        T = max(T, np.sqrt((10.0 / np.sqrt(3.0)) * D / lim.a_max))
    if lim.j_max:
        T = max(T, np.cbrt(60.0 * D / lim.j_max))
    return max(T, lim.min_duration)


def quintic_bridge(q0: np.ndarray, q1: np.ndarray, fps: float,
                   lim: BridgeLimits = BridgeLimits(), T_b: float | None = None) -> tuple[np.ndarray, float]:
    """Return (trajectory (N, dof), T_b). N = round(T_b*fps); the last frame equals q1."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    T_b = bridge_duration(q0, q1, lim) if T_b is None else T_b
    n = max(2, int(round(T_b * fps)) + 1)
    u = np.linspace(0.0, 1.0, n)
    s = quintic_s(u)[:, None]
    return q0[None, :] + s * (q1 - q0)[None, :], T_b
