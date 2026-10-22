"""Synthetic BVH + WAV + TSV "dataset" to exercise the full benchmark path when no public data
is reachable: Mixamo/BEAT-style skeleton, smooth random arm/spine motion whose strokes are
placed on a click track, so rhythm metrics have ground truth (BC / BAS should beat chance).

usage: python benchmark/make_synthetic.py [out_dir] [n_files] [seconds]
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.tests_roundtrip import synthetic_bvh  # noqa: E402
from trabot_a3.paths import OUT  # noqa: E402


def smooth_noise(T, fps, rng, amp, cutoff=1.5):
    x = rng.normal(size=T)
    F = np.fft.rfft(x); f = np.fft.rfftfreq(T, 1 / fps); F[f > cutoff] = 0
    y = np.fft.irfft(F, n=T); return amp * y / (np.abs(y).max() + 1e-9)


def main(out_dir: Path, n_files: int = 3, seconds: float = 20.0, fps: float = 60.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    T = int(seconds * fps)
    for k in range(n_files):
        beats = np.sort(rng.uniform(0.5, seconds - 0.5, int(seconds * 1.2)))   # ~1.2 strokes / s
        beats = beats[np.r_[True, np.diff(beats) > 0.35]]
        t = np.arange(T) / fps
        stroke = np.zeros(T)
        for b in beats:                       # a stroke: raise then drop around each beat (apex at the beat)
            stroke += 25 * np.exp(-((t - b) / 0.12) ** 2)
        poses = []
        base = {j: smooth_noise(T, fps, rng, a) for j, a in
                (("LeftArm_z", 35), ("LeftArm_x", 25), ("LeftForeArm_x", 40), ("RightArm_z", 35), ("RightArm_x", 25),
                 ("RightForeArm_x", 40), ("Spine1_x", 6), ("Spine1_y", 8), ("LeftHand_x", 15), ("RightHand_x", 15))}
        for i in range(T):
            s = stroke[i]
            poses.append({
                "LeftArm": (40 + base["LeftArm_z"][i] + 0.6 * s, base["LeftArm_x"][i] - 0.4 * s, 0),
                "LeftForeArm": (0, -60 + base["LeftForeArm_x"][i] - 0.8 * s, 0),
                "RightArm": (-40 - base["RightArm_z"][i] - 0.6 * s, base["RightArm_x"][i] - 0.4 * s, 0),
                "RightForeArm": (0, -60 + base["RightForeArm_x"][i] - 0.8 * s, 0),
                "LeftHand": (0, base["LeftHand_x"][i], 0), "RightHand": (0, base["RightHand_x"][i], 0),
                "Spine1": (0, base["Spine1_x"][i], base["Spine1_y"][i]),
            })
        stem = out_dir / f"synthetic_{k:02d}"
        synthetic_bvh(str(stem.with_suffix(".bvh")), poses, fps)
        # click track + pink-ish noise, 16 kHz mono 16-bit
        sr = 16000
        y = 0.02 * rng.normal(size=int(seconds * sr))
        for b in beats:
            i0 = int(b * sr); n = int(0.03 * sr)
            y[i0:i0 + n] += 0.8 * np.sin(2 * np.pi * 1000 * np.arange(n) / sr) * np.exp(-np.arange(n) / (0.008 * sr))
        with wave.open(str(stem.with_suffix(".wav")), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
        with open(stem.with_suffix(".tsv"), "w") as f:
            for i, b in enumerate(beats):
                f.write(f"{b:.3f}\t{b + 0.25:.3f}\tword{i}\n")
    print(f"wrote {n_files} synthetic takes ({seconds}s each) to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "synthetic_dataset",
         int(sys.argv[2]) if len(sys.argv) > 2 else 3, float(sys.argv[3]) if len(sys.argv) > 3 else 20.0)
