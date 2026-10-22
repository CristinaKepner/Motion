"""Online budgeted planner (TRABot §III-C, Eq. 11-12, Algorithm 1 planning step).

J(A_L; q) = C_tr(q, a_1) + Σ d_k + Σ C_tr(a_k, a_{k+1}) + C_ret(a_L) + G          (Eq. 11)
max L  s.t.  φ(a_k) = f_k  (k=1..L),  J(A_L; q) ≤ B(T_hat) = 3 T_hat               (Eq. 12)

Layered DP (report's derivation of the paper's Algorithm 1 inner step):
  D_1(i) = C_tr(q, a_i) + d_i
  D_k(i) = d_i + min_{j ∈ A_{k-1}} [ D_{k-1}(j) + C_tr(a_j, a_i) ]
  layer k feasible  iff  min_i [ D_k(i) + C_ret(a_i) + G ] ≤ B_rem
L* = deepest feasible layer. A deterministic beam (width 32) then picks the concrete
sequence of length L*, preferring fuller budget use, fewer recent repeats and L/R balance.
The DP's own argmin path is kept as a guaranteed-feasible fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atoms import AtomCatalog, MotionAtom
from .bridge import bridge_duration, BridgeLimits
from .data import NEUTRAL_RAD, CTRL_IDX


@dataclass
class PlannerParams:
    budget_mult: float = 3.0      # B = 3 T_hat
    guard_s: float = 0.35         # G
    beam_width: int = 32
    w_budget_use: float = 1.0     # prefer sequences that use more of the budget
    w_repeat: float = 0.6         # penalty per reuse of a recently played atom
    w_balance: float = 0.3        # penalty for one-sided hand usage
    recent_window: int = 6
    force_first: bool = False
    bridge: BridgeLimits = field(default_factory=BridgeLimits)


@dataclass
class PlanResult:
    atoms: list[MotionAtom]
    functions_requested: list[str]
    functions_planned: list[str]
    L_star: int
    budget: float
    cost: float
    cost_breakdown: dict
    layer_min_cost: list[float]
    beam_candidates: int
    fallback_used: bool
    truncated_reason: str | None

    def summary(self) -> str:
        done = " → ".join(self.functions_planned) or "(none)"
        return (f"L*={self.L_star}/{len(self.functions_requested)}  budget={self.budget:.2f}s  "
                f"J={self.cost:.2f}s  [{done}]" + (f"  stopped: {self.truncated_reason}" if self.truncated_reason else ""))


class Planner:
    def __init__(self, catalog: AtomCatalog, params: PlannerParams | None = None):
        self.cat = catalog
        self.p = params or PlannerParams()
        self._tr_cache: dict[tuple[str, str], float] = {}
        self.neutral = NEUTRAL_RAD[CTRL_IDX]

    # ---- cost model (shared by planner and executor: "规划与执行共用成本口径")
    def c_tr_pose(self, q_arm: np.ndarray, atom: MotionAtom) -> float:
        return bridge_duration(q_arm, atom.q_in, self.p.bridge)

    def c_tr(self, a: MotionAtom, b: MotionAtom) -> float:
        k = (a.atom_id, b.atom_id)
        if k not in self._tr_cache:
            self._tr_cache[k] = bridge_duration(a.q_out, b.q_in, self.p.bridge)
        return self._tr_cache[k]

    def c_ret(self, a: MotionAtom) -> float:
        return bridge_duration(a.q_out, self.neutral, self.p.bridge)

    def total_cost(self, q_arm: np.ndarray, seq: list[MotionAtom]) -> tuple[float, dict]:
        if not seq:
            return 0.0, {"enter": 0, "atoms": 0, "transitions": 0, "return": 0, "guard": 0}
        enter = self.c_tr_pose(q_arm, seq[0])
        atoms = sum(a.duration for a in seq)
        trans = sum(self.c_tr(seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        ret = self.c_ret(seq[-1])
        J = enter + atoms + trans + ret + self.p.guard_s
        return J, {"enter": round(enter, 3), "atoms": round(atoms, 3), "transitions": round(trans, 3),
                   "return": round(ret, 3), "guard": self.p.guard_s}

    # ---- Eq. 12 via layered DP + beam
    def plan(self, q_arm: np.ndarray, functions: list[str], budget_s: float,
             recent_ids: list[str] | None = None) -> PlanResult:
        recent = list(recent_ids or [])[-self.p.recent_window:]
        layers = [self.cat.by_label(f) for f in functions]
        B = budget_s
        # DP
        D: list[dict[str, float]] = []
        back: list[dict[str, str | None]] = []
        layer_min: list[float] = []
        L_star = 0
        reason = None
        for k, cands in enumerate(layers):
            if not cands:
                reason = f"no catalog atom for '{functions[k]}'"
                break
            Dk, Bk = {}, {}
            for a in cands:
                if k == 0:
                    Dk[a.atom_id] = self.c_tr_pose(q_arm, a) + a.duration
                    Bk[a.atom_id] = None
                else:
                    best, arg = float("inf"), None
                    for b in layers[k - 1]:
                        if b.atom_id in D[k - 1]:
                            c = D[k - 1][b.atom_id] + self.c_tr(b, a)
                            if c < best:
                                best, arg = c, b.atom_id
                    Dk[a.atom_id] = a.duration + best
                    Bk[a.atom_id] = arg
            full = {aid: Dk[aid] + self.c_ret(self._atom(aid)) + self.p.guard_s for aid in Dk}
            m = min(full.values())
            layer_min.append(round(m, 3))
            if m <= B + 1e-9:
                D.append(Dk); back.append(Bk); L_star = k + 1
            else:
                reason = f"budget exhausted at '{functions[k]}' (min J={m:.2f}s > B={B:.2f}s)"
                break
        if L_star == 0:
            if self.p.force_first and layers and layers[0]:
                a = min(layers[0], key=lambda x: self.c_tr_pose(q_arm, x) + x.duration + self.c_ret(x))
                J, br = self.total_cost(q_arm, [a])
                return PlanResult([a], functions, [a.label], 1, B, J, br, layer_min, 0, True,
                                  "force-first: over budget by design")
            return PlanResult([], functions, [], 0, B, 0.0, {}, layer_min, 0, False, reason)

        # DP fallback path (argmin at layer L*)
        last = min(D[L_star - 1], key=lambda aid: D[L_star - 1][aid] + self.c_ret(self._atom(aid)))
        fb = [last]
        for k in range(L_star - 1, 0, -1):
            fb.append(back[k][fb[-1]])
        fallback = [self._atom(x) for x in fb[::-1]]

        # backward DP: R_k(i) = exact minimum cost-to-go from atom i at layer k to the end of
        # the L*-prefix (incl. return + guard). Used as an admissible prune so every partial
        # beam path can still be completed within budget (report's recommendation, p.15).
        R: list[dict[str, float]] = [dict() for _ in range(L_star)]
        for a in layers[L_star - 1]:
            R[L_star - 1][a.atom_id] = self.c_ret(a) + self.p.guard_s
        for k in range(L_star - 2, -1, -1):
            for a in layers[k]:
                R[k][a.atom_id] = min(self.c_tr(a, b) + b.duration + R[k + 1][b.atom_id] for b in layers[k + 1])

        # beam search over concrete sequences of length L*
        beams: list[tuple[list[MotionAtom], float]] = [([], 0.0)]
        n_cand = 0
        for k in range(L_star):
            nxt = []
            for seq, c in beams:
                for a in layers[k]:
                    step = (self.c_tr_pose(q_arm, a) if k == 0 else self.c_tr(seq[-1], a)) + a.duration
                    nxt.append((seq + [a], c + step)); n_cand += 1
            nxt = [(s, c) for s, c in nxt if c + R[k][s[-1].atom_id] <= B + 1e-9]
            nxt.sort(key=lambda sc: self._score(sc[0], sc[1], B, recent))
            beams = nxt[: self.p.beam_width]
            if not beams:
                break
        fallback_used = False
        if beams and len(beams[0][0]) == L_star:
            seq = beams[0][0]
        else:
            seq, fallback_used = fallback, True
        J, br = self.total_cost(q_arm, seq)
        return PlanResult(seq, functions, [a.label for a in seq], L_star, B, J, br, layer_min,
                          n_cand, fallback_used, reason if L_star < len(functions) else None)

    def _score(self, seq: list[MotionAtom], partial_cost: float, B: float, recent: list[str]) -> float:
        # lower is better. Fuller budget use rewarded; repeats and one-sidedness penalised.
        use = partial_cost / max(B, 1e-9)
        reps = sum(1 for a in seq if a.atom_id in recent) + (len(seq) - len({a.atom_id for a in seq}))
        hands = [a.hand for a in seq]
        l, r = hands.count("L"), hands.count("R")
        imbalance = abs(l - r) / max(len(seq), 1)
        return -self.p.w_budget_use * use + self.p.w_repeat * reps + self.p.w_balance * imbalance

    def _atom(self, aid: str) -> MotionAtom:
        for a in self.cat.atoms:
            if a.atom_id == aid:
                return a
        raise KeyError(aid)
