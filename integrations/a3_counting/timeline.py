"""Audio-clock gesture scheduling with explicit preparation and hold windows."""

from dataclasses import dataclass
import math

import numpy as np

from .hand import preset
from .o10 import PREPARATION, SETTLE


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t*t*t*(10 + t*(-15+6*t))


@dataclass(frozen=True)
class CountEvent:
    count: int
    onset: float
    end: float


class Timeline:
    def __init__(self, events, duration, preparation=PREPARATION, settle=SETTLE, release_hold=.35):
        self.events = [CountEvent(**{k: e[k] for k in ("count", "onset", "end")}) for e in events]
        self.duration = duration
        self.preparation = preparation
        self.settle = settle
        self.release = duration + release_hold
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("timeline needs a positive finite duration")
        if not all(math.isfinite(x) and x >= 0 for x in (preparation, settle, release_hold)) or preparation == 0:
            raise ValueError("invalid preparation/settling/hold durations")
        previous_end = 0
        for event in self.events:
            preset(event.count)
            if event.count == 0 or not 0 <= event.onset < event.end <= duration:
                raise ValueError("count events must be within the audio duration")
            if event.onset-preparation-settle < previous_end:
                raise ValueError("speech events are too close to hold each word and prepare the next gesture")
            previous_end = event.end

    def sample(self, audio_time):
        """Return targets and phase, using the actual played-audio time in seconds."""
        if not math.isfinite(audio_time):
            raise ValueError("non-finite audio clock")
        previous = preset(0)
        count = 0
        for event in self.events:
            start = event.onset-self.preparation-self.settle
            if audio_time < start:
                return previous, f"hold_{count}" if count else "ready"
            target = preset(event.count)
            if audio_time < event.onset-self.settle:
                weight = smoothstep((audio_time-start)/self.preparation)
                return previous+(target-previous)*weight, f"prepare_{event.count}"
            previous, count = target, event.count
        if audio_time < self.release:
            return previous, f"hold_{count}"
        weight = smoothstep((audio_time-self.release)/self.preparation)
        return previous+(preset(0)-previous)*weight, "release" if weight < 1 else "done"

    def windows(self):
        return [{"event": f"count_{e.count}", "prepare_at": e.onset-self.preparation-self.settle,
                 "target_at": e.onset-self.settle, "word_onset": e.onset, "word_end": e.end,
                 "hold_until": (self.events[i+1].onset-self.preparation-self.settle
                                if i+1 < len(self.events) else self.release)}
                for i, e in enumerate(self.events)]
