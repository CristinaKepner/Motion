"""Streaming hand/body fusion agent.

Input: a stream of timestamped events (speech tokens, game cues, operator commands) arriving at wall-clock
time `t_arrive`, each carrying the time the gesture *should* land (`t_target`, e.g. the stressed syllable or
the "布!" beat). Output: one fused timeline — body (31 joints, from the TRABot atom library) + left hand +
right hand (10 O10 channels each) — that is sampled by a 30 Hz tick, or seeked to any time (`at(t)`).

Response-time model (all measured, all logged per event):
    t_arrive  → decide (planning latency, measured)
              → earliest start  = t_arrive + latency + bridge_min
              → scheduled start = max(earliest, t_target − stroke_lead)       (stroke lands on t_target)
              → if earliest > t_target − stroke_lead the stroke is late: late_ms is recorded,
                 and tempo is raised (up to tempo_max) to claw the delay back.
Fusion rules: each hand is one track; a new event on a hand pre-empts the running atom at the current pose
(the new atom's prepare keyframe starts from the live pose, so there is never a jump). Body atoms are
scheduled by the TRABot planner/bridge when a body function is attached to the event.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .bridge import quintic_s, bridge_duration
from .data import NEUTRAL_RAD
from .hand import HandAtom, HandStyle, HAND_PRESETS, NEUTRAL_HAND, POSES, ZH_COUNT, make_atom, HandModel

FPS = 30.0


@dataclass
class Event:
    t_arrive: float
    kind: str                      # "hand" | "body" | "both"
    hand_label: str | None = None  # key of POSES
    side: str = "R"                # "L" | "R" | "LR"
    t_target: float | None = None  # when the stroke should land; None = as soon as possible
    style: str = "neutral"
    hold_s: float = 0.45
    body_label: str | None = None  # semantic function for the body layer (via atom library), optional
    text: str = ""
    event_id: str = ""


@dataclass
class Placed:
    event_id: str
    side: str
    atom: HandAtom
    t_start: float
    t_end: float
    latency_ms: float
    late_ms: float
    tempo_boost: float
    t_target: float | None
    text: str

    def meta(self):
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "atom"}
        d["atom"] = self.atom.meta()
        return d


class HandTrack:
    """one hand: list of placed atoms, non-overlapping after pre-emption; pose at any t by sampling."""

    def __init__(self, side: str):
        self.side = side
        self.items: list[Placed] = []

    def pose_at(self, t: float) -> np.ndarray:
        cur = NEUTRAL_HAND.copy()
        for p in self.items:
            if t >= p.t_start:
                cur = p.atom.sample(min(t, p.t_end) - p.t_start) if t < p.t_end else p.atom.sample(p.atom.duration)
            else:
                break
        return cur

    def truncate(self, t: float):
        """pre-empt whatever is running at t: later items dropped, running item cut at t."""
        keep = []
        for p in self.items:
            if p.t_start >= t:
                continue
            if p.t_end > t:
                p.t_end = t
            keep.append(p)
        self.items = keep


class FusionAgent:
    def __init__(self, bridge_min_s: float = 0.08, stroke_lead_s: float | None = None, tempo_max: float = 1.8,
                 body_library=None):
        self.bridge_min = bridge_min_s
        self.stroke_lead = stroke_lead_s
        self.tempo_max = tempo_max
        self.hands = {"L": HandTrack("L"), "R": HandTrack("R")}
        self.body_keys: list[tuple[float, np.ndarray]] = [(0.0, NEUTRAL_RAD.copy())]  # body keyframes (t, q31)
        self.body_library = body_library
        self.log: list[dict] = []
        self.hm = HandModel()

    # ---------------------------------------------------------------- scheduling
    def submit(self, ev: Event) -> list[Placed]:
        t0 = time.perf_counter()
        st = HAND_PRESETS.get(ev.style, HAND_PRESETS["neutral"])
        placed = []
        sides = ([] if ev.kind == "body" or ev.hand_label is None else ["L", "R"] if ev.side == "LR" else [ev.side])
        for side in sides:
            atom = make_atom(ev.hand_label, side, start=self.hands[side].pose_at(ev.t_arrive), st=st, hold_s=ev.hold_s)
            # time from atom start to the end of the stroke keyframe
            stroke_end = next(k.t for k in atom.keys if k.phase == "stroke")
            latency = (time.perf_counter() - t0) * 1000
            earliest = ev.t_arrive + latency / 1000 + self.bridge_min
            lead = stroke_end if self.stroke_lead is None else self.stroke_lead
            boost, late = 1.0, 0.0
            if ev.t_target is None:
                t_start = earliest
            else:
                t_start = ev.t_target - lead
                if t_start < earliest:
                    # late: compress the stroke to land as close to target as the tempo cap allows
                    need = (ev.t_target - earliest)
                    boost = float(np.clip(stroke_end / max(need, 1e-3), 1.0, self.tempo_max))
                    st2 = HandStyle(**{**st.__dict__, "tempo": st.tempo * boost})
                    atom = make_atom(ev.hand_label, side, start=self.hands[side].pose_at(earliest), st=st2, hold_s=ev.hold_s)
                    stroke_end = next(k.t for k in atom.keys if k.phase == "stroke")
                    t_start = earliest
                    late = max(0.0, (t_start + stroke_end - ev.t_target) * 1000)
            self.hands[side].truncate(t_start)
            atom.annotation.update({"event_id": ev.event_id, "text": ev.text, "t_arrive": ev.t_arrive, "t_target": ev.t_target})
            p = Placed(ev.event_id, side, atom, t_start, t_start + atom.duration, latency, late, boost, ev.t_target, ev.text)
            self.hands[side].items.append(p)
            self.hands[side].items.sort(key=lambda x: x.t_start)
            placed.append(p)
            self.log.append({"event": ev.__dict__ | {"side": side}, "t_start": round(t_start, 4), "stroke_lands": round(t_start + stroke_end, 4),
                             "latency_ms": round(latency, 3), "late_ms": round(late, 1), "tempo_boost": round(boost, 3),
                             "response_ms": round((t_start + stroke_end - ev.t_arrive) * 1000, 1)})
        if ev.body_label and self.body_library is not None:
            self._schedule_body(ev, placed[0].t_start if placed else ev.t_arrive + self.bridge_min)
        return placed

    def _schedule_body(self, ev: Event, t_start: float):
        """body layer: pick the library atom for the function, bridge from the current body pose (TRABot Eq.13–14)."""
        atom = self.body_library.get(ev.body_label)
        if atom is None:
            return
        frames = np.asarray(atom.q if hasattr(atom, "q") else atom)   # MotionAtom or (T,31) array
        q_now = self.body_at(t_start)
        tb = max(bridge_duration(q_now, frames[0]), 0.15)
        self.body_keys = [k for k in self.body_keys if k[0] < t_start]
        self.body_keys.append((t_start, q_now))
        t = t_start + tb
        for i, q in enumerate(frames):
            self.body_keys.append((t + i / FPS, q))
        self.body_keys.sort(key=lambda k: k[0])

    # ---------------------------------------------------------------- playback / seek
    def body_at(self, t: float) -> np.ndarray:
        ks = self.body_keys
        if t <= ks[0][0]:
            return ks[0][1].copy()
        if t >= ks[-1][0]:
            return ks[-1][1].copy()
        i = int(np.searchsorted([k[0] for k in ks], t, side="right") - 1)
        (ta, qa), (tb, qb) = ks[i], ks[i + 1]
        w = (t - ta) / max(tb - ta, 1e-6)
        if tb - ta > 2.5 / FPS:   # a bridge gap → quintic (Eq.13)
            w = float(quintic_s(np.array([w]))[0])
        return qa + w * (qb - qa)

    def at(self, t: float) -> dict:
        """seek: full fused state at absolute time t (e.g. at(5.0) = '回放第五秒')."""
        L, Rh = self.hands["L"].pose_at(t), self.hands["R"].pose_at(t)
        active = {s: next((p.event_id for p in self.hands[s].items if p.t_start <= t < p.t_end), None) for s in "LR"}
        return {"t": t, "body": self.body_at(t), "L": L, "R": Rh, "L_o10": self.hm.to_o10(L), "R_o10": self.hm.to_o10(Rh), "active": active}

    def ticks(self, t_end: float, fps: float = FPS):
        """streaming generator: 30 Hz frames in order (what the robot bridge would consume)."""
        n = int(np.ceil(t_end * fps))
        for i in range(n + 1):
            yield self.at(i / fps)

    @property
    def t_end(self) -> float:
        te = max([p.t_end for s in "LR" for p in self.hands[s].items] + [self.body_keys[-1][0]])
        return te + 0.3

    def timeline(self) -> dict:
        return {"L": [p.meta() for p in self.hands["L"].items], "R": [p.meta() for p in self.hands["R"].items],
                "events": self.log, "t_end": round(self.t_end, 3),
                "response_ms": {"mean": round(float(np.mean([e["response_ms"] for e in self.log])), 1) if self.log else None,
                                "max_late_ms": round(max([e["late_ms"] for e in self.log], default=0.0), 1),
                                "plan_latency_ms_mean": round(float(np.mean([e["latency_ms"] for e in self.log])), 3) if self.log else None}}


# ---------------------------------------------------------------- scripted scenarios
def script_one_plus_one(agent: FusionAgent, t0: float = 0.5, style: str = "neutral", beat: float = 0.9) -> list[Event]:
    """「1 + 1 = 2」: left hand shows 1, right hand shows 1, both hands meet (plus), both show 2."""
    evs = [
        Event(t0 - 0.35, "hand", "one", "L", t_target=t0, style=style, text="一", event_id="e1"),
        Event(t0 + beat - 0.35, "hand", "one", "R", t_target=t0 + beat, style=style, text="加一", event_id="e2"),
        Event(t0 + 2 * beat - 0.35, "hand", "flat", "LR", t_target=t0 + 2 * beat, style=style, text="等于", event_id="e3", hold_s=0.3),
        Event(t0 + 3 * beat - 0.35, "hand", "two", "LR", t_target=t0 + 3 * beat, style=style, text="二", event_id="e4", hold_s=0.9),
    ]
    for e in evs:
        agent.submit(e)
    return evs


def script_rps(agent: FusionAgent, t0: float = 0.5, choice: str = "scissors", style: str = "lively", beat: float = 0.6,
               side: str = "R") -> list[Event]:
    """「石头剪刀布」: three fist pumps on the beat, reveal on the fourth beat."""
    evs = []
    for i, word in enumerate(["石头", "剪刀", "布"]):
        evs.append(Event(t0 + i * beat - 0.3, "hand", "rock", side, t_target=t0 + i * beat, style=style, hold_s=0.05, text=word, event_id=f"rps{i}"))
    evs.append(Event(t0 + 3 * beat - 0.3, "hand", choice, side, t_target=t0 + 3 * beat, style=style, hold_s=1.2, text=f"出{POSES[choice]['zh']}", event_id="rps_reveal"))
    for e in evs:
        agent.submit(e)
    return evs


def script_count(agent: FusionAgent, n: int = 5, t0: float = 0.5, beat: float = 0.7, style: str = "neutral", side: str = "R") -> list[Event]:
    evs = [Event(t0 + i * beat - 0.3, "hand", ZH_COUNT[i + 1], side, t_target=t0 + i * beat, style=style, hold_s=0.3, text=str(i + 1), event_id=f"c{i+1}") for i in range(n)]
    for e in evs:
        agent.submit(e)
    return evs


def save_timeline(agent: FusionAgent, path):
    path.write_text(json.dumps(agent.timeline(), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
