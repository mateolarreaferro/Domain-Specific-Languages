#!/usr/bin/env python3
"""sonicbin.py  –  *Hear* your data
===================================
Turn **any** binary file into a MIDI clip you can drag straight into your DAW.
The API is tiny so you can tweak mappings in pure Python, yet it hides all the
low‑level MIDI plumbing.

Quick CLI usage
---------------
```bash
pip install mido           # one‑time dependency (pure‑Python)
python sonicbin.py firmware.bin               # ⇒ firmware.mid
python sonicbin.py picture.jpg out.mid -t 90  # custom name & tempo
python sonicbin.py file.bin --merge           # single‑track output
```

Quick API usage
---------------
```python
from sonicbin import Sonifier, default_mapping

Sonifier("file.bin", tempo_bpm=100, merge_tracks=True) \
    .add_mapping(default_mapping())                 \
    .write_midi("file.mid")
```

Key concepts
------------
* **Sonifier** – loads the binary, sets tempo/PPQ, exports MIDI.
* **ByteMapping** – lambdas that map a byte (and/or its index) to *pitch*,
  *velocity*, *duration* and *channel*.  Add as many as you like.
* **slice_filter(slice)** – helper to target header vs. payload etc.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

# ────────────────────────────────────────────────────────────── MIDI backend
try:
    from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
except ImportError as exc:  # pragma: no cover – shown to user when missing
    raise ImportError("Install the *mido* package:  pip install mido") from exc

# ────────────────────────────────────────────────────────────── Small helpers
PopFunc = Callable[[int], int]                      # byte → value
ChannelFunc = Callable[[int], int] | Callable[[int, int], int]


def popcount(x: int) -> int:
    """Number of 1‑bits in *x* (0–255)."""
    return bin(x & 0xFF).count("1")


def slice_filter(slc: slice) -> Callable[[int], bool]:
    """Return a predicate selecting byte *indices* inside *slc* (step honoured)."""
    start, stop, step = slc.start or 0, slc.stop, slc.step or 1
    if stop is None:  # open‑ended slice (e.g. 32:)
        return lambda idx: idx >= start and (idx - start) % step == 0
    return lambda idx: start <= idx < stop and (idx - start) % step == 0

# ────────────────────────────────────────────────────────────── Core classes
class ByteMapping:
    """Map each selected byte to a MIDI note.

    Supply lambdas for *pitch*, *velocity*, *duration_ticks*, *channel*.
    The optional *byte_filter* lets you restrict the mapping to specific
    indices (header vs. payload, every Nth byte, etc.).
    """

    def __init__(
        self,
        name: str,
        pitch: PopFunc,
        velocity: PopFunc,
        duration_ticks: PopFunc,
        channel: ChannelFunc,
        *,
        byte_filter: Optional[Callable[[int], bool]] = None,
    ) -> None:
        self.name = name
        self.pitch_f = pitch
        self.vel_f = velocity
        self.dur_f = duration_ticks
        self.chan_f = channel
        self.filter = byte_filter or (lambda _i: True)

    def map(self, idx: int, byte: int) -> Optional[Tuple[int, int, int, int]]:
        """Return (pitch, velocity, duration, channel) or *None* if skipped."""
        if not self.filter(idx):
            return None
        pitch = max(0, min(127, self.pitch_f(byte)))
        velocity = max(1, min(127, self.vel_f(byte)))
        duration = max(1, self.dur_f(byte))
        try:
            chan = self.chan_f(idx, byte)
        except TypeError:  # mapping only needs index
            chan = self.chan_f(idx)
        return pitch, velocity, duration, chan % 16


class Sonifier:
    """High‑level façade: load → map → write .mid"""

    def __init__(
        self,
        binary_path: Union[str, Path],
        *,
        tempo_bpm: int = 120,
        ticks_per_beat: int = 480,
        merge_tracks: bool = False,
    ) -> None:
        self.path = Path(binary_path)
        if not self.path.is_file():
            raise FileNotFoundError(binary_path)
        self.data: bytes = self.path.read_bytes()
        self.tempo_bpm = tempo_bpm
        self.tpb = ticks_per_beat
        self.merge = merge_tracks
        self.mappings: List[ByteMapping] = []

    # –– DSL verb ––
    def add_mapping(self, mapping: ByteMapping) -> "Sonifier":
        self.mappings.append(mapping)
        return self

    # –– Driver ––
    def write_midi(self, out_path: Union[str, Path]) -> Path:
        if not self.mappings:
            raise RuntimeError("Add at least one ByteMapping before writing MIDI.")

        # Type‑0 SMF if merging (single track w/all events), else type‑1
        mf = MidiFile(type=0 if self.merge else 1, ticks_per_beat=self.tpb)

        # Track 0 – tempo/meta
        meta = MidiTrack(); mf.tracks.append(meta)
        meta.append(MetaMessage("set_tempo", tempo=bpm2tempo(self.tempo_bpm), time=0))

        if self.merge:
            tracks = [meta] * 16  # every channel uses track 0
        else:
            tracks = [MidiTrack() for _ in range(16)]
            mf.tracks.extend(tracks)

        for idx, byte in enumerate(self.data):
            for m in self.mappings:
                mapped = m.map(idx, byte)
                if mapped is None:
                    continue
                pitch, vel, dur, ch = mapped
                if self.merge:
                    ch = 0  # keep DAWs from splitting channels
                tr = tracks[ch]
                tr.append(Message("note_on",  channel=ch, note=pitch, velocity=vel, time=0))
                tr.append(Message("note_off", channel=ch, note=pitch, velocity=0,  time=dur))

        mf.save(str(out_path))
        return Path(out_path).resolve()

# ────────────────────────────────────────────────────────────── Ready‑made mapping

def _d_pitch(b: int) -> int:   # C1–C8-ish sweep
    return 24 + int(b / 255 * 84)

def _d_velocity(b: int) -> int:
    return 40 + ((b >> 1) & 0x3F)

def _d_duration(b: int) -> int:
    return 120 + popcount(b) * 30

def _d_channel(idx: int) -> int:
    return (idx // 4) % 8

def default_mapping(name: str = "default") -> ByteMapping:
    """Handy starter mapping covering pitch/velocity/duration/channel."""
    return ByteMapping(name, _d_pitch, _d_velocity, _d_duration, _d_channel)

# ────────────────────────────────────────────────────────────── CLI entry‑point
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Binary → Standard MIDI File")
    ap.add_argument("binary", help="Path to the binary file")
    ap.add_argument("out_midi", nargs="?", help="Output .mid (defaults to <input>.mid)")
    ap.add_argument("--tempo", "-t", type=int, default=120, help="Tempo (BPM)")
    ap.add_argument("--ticks", "-ppq", type=int, default=480, help="PPQ resolution")
    ap.add_argument("--merge", action="store_true", help="Single‑track output")
    args = ap.parse_args()

    out_name = args.out_midi or f"{Path(args.binary).with_suffix('').name}.mid"

    Sonifier(args.binary,
             tempo_bpm=args.tempo,
             ticks_per_beat=args.ticks,
             merge_tracks=args.merge)                 \
        .add_mapping(default_mapping())               \
        .write_midi(out_name)

    print(f"✓  Wrote {out_name}")
