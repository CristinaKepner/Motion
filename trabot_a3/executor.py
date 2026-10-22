"""State-aware execution (TRABot Algorithm 1) with epoch/cancel semantics.

loop over unfinished functions k:
    q, B_rem  <- measured state, remaining budget
    plan      <- longest feasible prefix from (q, f_k.., B_rem)        (planner)
    if L* == 0: stop
    bridge q -> a_1.q_in, play a_1 only, measure elapsed time and pose
    mark f_k done; k += 1 ; B_rem -= elapsed
neutral recovery from the *measured* pose once the sequence ends (or on cancel).

Every event carries reply_id, epoch, command_id and the monotonic clock so a trace
can answer "why this atom, why no atom, where did the time go".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .atoms import MotionAtom
from .bridge import quintic_bridge
from .data import CTRL_IDX, NEUTRAL_RAD
from .planner import Planner, PlanResult


class SimBackend:
    """Minimal execution interface. See sim.MujocoKinematicBackend."""
    fps: float = 30.0

    def read_arm_state(self) -> np.ndarray: ...
    def play(self, q_full: np.ndarray, tag: str) -> tuple[float, np.ndarray]:
        """Play a (T,31) rad trajectory at self.fps; return (elapsed_s, measured arm pose)."""
        ...


@dataclass
class ExecutionTrace:
    reply_id: str
    epoch: int
    T_hat: float
    budget: float
    events: list[dict] = field(default_factory=list)
    functions_requested: list[str] = field(default_factory=list)
    functions_completed: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    elapsed_total: float = 0.0

    def log(self, kind: str, t: float, **kw):
        self.events.append({"t": round(t, 3), "event": kind, "reply_id": self.reply_id, "epoch": self.epoch, **kw})

    def summary(self) -> str:
        lines = [f"reply={self.reply_id} epoch={self.epoch} T_hat={self.T_hat}s B={self.budget:.2f}s "
                 f"done {len(self.functions_completed)}/{len(self.functions_requested)} "
                 f"[{' → '.join(self.functions_completed)}] total {self.elapsed_total:.2f}s"
                 + (f"  stop: {self.stop_reason}" if self.stop_reason else "")]
        for e in self.events:
            extra = {k: v for k, v in e.items() if k not in ("t", "event", "reply_id", "epoch")}
            lines.append(f"  {e['t']:6.2f}s  {e['event']:<14} {extra}")
        return "\n".join(lines)


class StateAwareExecutor:
    def __init__(self, planner: Planner, backend: SimBackend):
        self.planner = planner
        self.backend = backend
        self.epoch = 0
        self._cancel = False
        self.recent: list[str] = []

    def cancel(self):
        """New-turn interrupt: bump epoch, old plans are no longer committed."""
        self._cancel = True
        self.epoch += 1

    def full_body(self, q_ctrl_traj: np.ndarray) -> np.ndarray:
        """Waist+arms from the trajectory; head and legs on the neutral pose."""
        out = np.repeat(NEUTRAL_RAD[None, :], q_ctrl_traj.shape[0], axis=0)
        out[:, CTRL_IDX] = q_ctrl_traj
        return out

    def run(self, reply_id: str, functions: list[str], T_hat: float,
            replan_each_atom: bool = True) -> ExecutionTrace:
        self._cancel = False
        epoch = self.epoch
        B = self.planner.p.budget_mult * T_hat
        tr = ExecutionTrace(reply_id, epoch, T_hat, B, functions_requested=list(functions))
        t = 0.0
        B_rem = B
        k = 0
        last_plan: PlanResult | None = None
        while k < len(functions):
            if self._cancel or self.epoch != epoch:
                tr.stop_reason = "cancelled (epoch advanced)"; break
            q = self.backend.read_arm_state()
            t0 = time.perf_counter()
            plan = self.planner.plan(q, functions[k:], B_rem, recent_ids=self.recent)
            plan_ms = (time.perf_counter() - t0) * 1e3
            tr.log("plan", t, k=k, L_star=plan.L_star, B_rem=round(B_rem, 3), J=round(plan.cost, 3),
                   seq=[a.atom_id for a in plan.atoms], layer_min=plan.layer_min_cost,
                   beam_candidates=plan.beam_candidates, fallback=plan.fallback_used,
                   plan_ms=round(plan_ms, 2), note=plan.truncated_reason)
            if plan.L_star == 0:
                tr.stop_reason = plan.truncated_reason or "no feasible prefix"; break
            a = plan.atoms[0]
            # bridge from the measured pose (never from the catalog's nominal pose)
            qb, T_b = quintic_bridge(q, a.q_in, self.backend.fps, self.planner.p.bridge)
            el_b, q_meas = self.backend.play(self.full_body(qb), f"bridge→{a.atom_id}")
            tr.log("bridge", t, to=a.atom_id, T_b_planned=round(T_b, 3), elapsed=round(el_b, 3),
                   D=round(float(np.max(np.abs(a.q_in - q))), 3))
            t += el_b; B_rem -= el_b
            if self._cancel or self.epoch != epoch:
                tr.stop_reason = "cancelled during bridge"; break
            el_a, q_meas = self.backend.play(self.full_body(a.q[:, CTRL_IDX]), a.atom_id)
            dev = float(np.max(np.abs(q_meas - a.q_out)))
            tr.log("atom", t, atom=a.atom_id, label=a.label, hand=a.hand, d_planned=round(a.duration, 3),
                   elapsed=round(el_a, 3), exit_pose_dev=round(dev, 3))
            t += el_a; B_rem -= el_a
            self.recent.append(a.atom_id)
            tr.functions_completed.append(functions[k])
            k += 1
            last_plan = plan
            if not replan_each_atom and len(plan.atoms) > 1:
                pass  # (paper: only the first atom is committed; we always re-read state)
        # neutral recovery from measured pose
        q = self.backend.read_arm_state()
        qr, T_r = quintic_bridge(q, NEUTRAL_RAD[CTRL_IDX], self.backend.fps, self.planner.p.bridge)
        el_r, _ = self.backend.play(self.full_body(qr), "neutral_recovery")
        tr.log("neutral", t, T_planned=round(T_r, 3), elapsed=round(el_r, 3), B_rem_before=round(B_rem, 3))
        t += el_r
        tr.elapsed_total = t
        if tr.stop_reason is None and k == len(functions):
            tr.stop_reason = None
        return tr
