"""Generate the fixed Chinese phrase and retain TTS word-boundary evidence."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import edge_tts
import imageio_ffmpeg

TEXT = "我们来数数：一、二、三。"


def space_fixed_counts(wav, events):
    """Insert PCM silence before count cues so O10 can prepare each gesture.

    Speech samples remain unchanged. Original TTS boundaries are retained in
    metadata; returned event times include the inserted silence.
    """
    import math
    from .o10 import LEAD
    with wave.open(str(wav)) as stream:
        params = stream.getparams()
        audio = stream.readframes(stream.getnframes())
    sr, width = params.framerate, params.sampwidth*params.nchannels
    chunks, shifted, insertions = [], [], []
    cursor = extra = 0
    previous_end = 0.
    for event in events:
        cut = round(event["onset"]*sr)
        gap = max(0, math.ceil((previous_end+LEAD+.02-event["onset"]-extra/sr)*sr))
        chunks.append(audio[cursor*width:cut*width])
        chunks.append(bytes(gap*width))
        cursor = cut
        extra += gap
        shifted.append({**event, "onset": event["onset"]+extra/sr, "end": event["end"]+extra/sr})
        previous_end = shifted[-1]["end"]
        if gap:
            insertions.append({"before_original_sample": cut, "silence_samples": gap})
    chunks.append(audio[cursor*width:])
    with wave.open(str(wav), "wb") as stream:
        stream.setparams(params)
        stream.writeframes(b"".join(chunks))
    return shifted, insertions


async def synthesize(output, text=TEXT, voice="zh-CN-XiaoxiaoNeural", rate="-20%", require_fixed_counts=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "speech.partial.mp3"
    boundaries = []
    with temporary.open("wb") as stream:
        async for part in edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary").stream():
            if part["type"] == "audio":
                stream.write(part["data"])
            elif part["type"] == "WordBoundary":
                boundaries.append(part)
    mp3 = output / "speech.mp3"
    temporary.replace(mp3)
    wav = output / "speech.wav"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ac", "1", "-ar", "24000", str(wav)], check=True)
    events = []
    for word in boundaries:
        if require_fixed_counts and word["text"] in ("一", "二", "三"):
            count = {"一": 1, "二": 2, "三": 3}[word["text"]]
            events.append({"name": f"count_{count}", "count": count,
                           "onset": word["offset"] / 1e7,
                           "end": (word["offset"]+word["duration"]) / 1e7})
    if require_fixed_counts and [e["count"] for e in events] != [1, 2, 3]:
        raise ValueError(f"Expected three distinct count word boundaries, got {events}")
    insertions = []
    if require_fixed_counts:
        events, insertions = space_fixed_counts(wav, events)
    with wave.open(str(wav)) as stream:
        duration = stream.getnframes()/stream.getframerate()
    metadata = {"text": text, "voice": voice, "rate": rate, "provider": "edge-tts",
                "boundary_source": "TTS WordBoundary + recorded PCM silence insertions; raw boundaries in 100 ns units",
                "duration": duration, "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "events": events, "raw_boundaries": boundaries, "pcm_silence_insertions": insertions}
    (output / "speech.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(events, ensure_ascii=False), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/counting/o10/fixed_audio"))
    args = parser.parse_args()
    asyncio.run(synthesize(args.output))


if __name__ == "__main__":
    main()
