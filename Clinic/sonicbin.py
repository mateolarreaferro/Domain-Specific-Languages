"""sonicbin.py – Internal DSL for *sonifying* binary files to MIDI
===================================================================
Hear byte‑level structure – drag the resulting `.mid` straight into a DAW.
This internal DSL stays minimal: you import **Sonifier** and declare one or
more **ByteMapping** objects (or use the built‑ins).  Everything else is just
regular Python, so you can slice, loop, or calculate however you like.

Quick start (run from your *Clinic* folder)
-------------------------------------------
```bash
python sonicbin.py test_cases/fixed_size_fields/test_no_encryption.fsf out.mid --tempo 110
```

Using in a mapping script
-------------------------
```python
# my_mapping.py (same folder as sonicbin.py)
from sonicbin import Sonifier, default_mapping

s = Sonifier("test_cases/linked_list_of_files/multiple_files.llf", tempo_bpm=90)
s.add_mapping(default_mapping()).write_midi("llf.mid")
```
Run it with `python my_mapping.py` and drop *llf.mid* into Ableton/Logic.

Field‑aware example
-------------------
```python
from sonicbin import Sonifier, ByteMapping, popcount, slice_filter
BIN = "test_cases/fixed_size_fields/test_no_encryption.fsf"
HEADER  = slice(0, 1)    # count byte
PAYLOAD = slice(1, None) # ascii payload

son = Sonifier(BIN)
son.add_mapping(ByteMapping(
        "hdr", pitch=lambda b: 36+b, velocity=lambda b: 120,
        duration_ticks=lambda b: 960, channel=lambda i: 0,
        byte_filter=slice_filter(HEADER)))
son.add_mapping(ByteMapping(
        "txt", pitch=lambda b: 60+(b%24), velocity=lambda b: 50+popcount(b)*8,
        duration_ticks=lambda b: 120, channel=lambda i: 1,
        byte_filter=slice_filter(PAYLOAD)))
son.write_midi("field_demo.mid")
```

---------------------------------------------------------------------
IMPLEMENTATION
---------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Union, Optional, Tuple

# ───────────── MIDI backend (mido) ─────────────
try:
    from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
except ImportError as exc:
    raise ImportError(
        "sonicbin requires the 'mido' package. Install it with\n"
        "    pip install mido\n"
        "(optionally add python-rtmidi if you ever need real‑time output)."
    ) from exc

# ───────────── Helpers ─────────────

def popcount(x: int) -> int:
    """Number of 1‑bits in *x* (0–255)."""
    return bin(x & 0xFF).count("1")


def slice_filter(slc: slice) -> Callable[[int], bool]:
    """Predicate that selects byte indexes inside *slc* (step supported)."""
    start, stop, step = slc.start or 0, slc.stop, slc.step or 1
    if stop is None:
        return lambda idx: idx >= start and (idx - start) % step == 0
    return lambda idx: start <= idx < stop and (idx - start) % step == 0

# ───────────── Core DSL types ─────────────
class ByteMapping:
    """Describe how selected bytes become MIDI notes."""

    def __init__(
        self,
        name: str,
        pitch: Callable[[int], int],
        velocity: Callable[[int], int],
        duration_ticks: Callable[[int], int],
        channel: Callable[[int], int] | Callable[[int, int], int],
        *,
        byte_filter: Optional[Callable[[int], bool]] = None,
    ):
        self.name = name
        self.pitch_f = pitch
        self.vel_f = velocity
        self.dur_f = duration_ticks
        self.chan_f = channel
        self.filter = byte_filter or (lambda idx: True)

    def map(self, index: int, byte: int) -> Optional[Tuple[int, int, int, int]]:
        if not self.filter(index):
            return None
        pitch = max(0, min(127, self.pitch_f(byte)))
        velocity = max(1, min(127, self.vel_f(byte)))
        duration = max(1, self.dur_f(byte))
        try:
            chan = self.chan_f(index, byte)
        except TypeError:
            chan = self.chan_f(index)
        chan = chan % 16
        return pitch, velocity, duration, chan


class Sonifier:
    """Turn a binary blob into a multi‑track Standard MIDI File."""

    def __init__(
        self,
        binary_path: Union[str, Path],
        *,
        tempo_bpm: int = 120,
        ticks_per_beat: int = 480,
    ):
        self.path = Path(binary_path)
        if not self.path.is_file():
            raise FileNotFoundError(binary_path)
        self.data: bytes = self.path.read_bytes()
        self.tempo_bpm = int(tempo_bpm)
        self.tpb = int(ticks_per_beat)
        self.mappings: List[ByteMapping] = []

    def add_mapping(self, mapping: ByteMapping) -> "Sonifier":
        self.mappings.append(mapping)
        return self

    def write_midi(self, out_path: Union[str, Path]):
        if not self.mappings:
            raise RuntimeError("No mappings registered – add at least one.")

        midi = MidiFile(ticks_per_beat=self.tpb)
        meta = MidiTrack(); midi.tracks.append(meta)
        meta.append(MetaMessage("set_tempo", tempo=bpm2tempo(self.tempo_bpm), time=0))
        tracks = [MidiTrack() for _ in range(16)]; midi.tracks.extend(tracks)

        for idx, byte in enumerate(self.data):
            for m in self.mappings:
                packed = m.map(idx, byte)
                if packed is None:
                    continue
                pitch, vel, dur, ch = packed
                tr = tracks[ch]
                tr.append(Message("note_on",  channel=ch, note=pitch, velocity=vel, time=0))
                tr.append(Message("note_off", channel=ch, note=pitch, velocity=0,  time=dur))

        midi.save(str(out_path))
        return Path(out_path).resolve()

# ───────────── Default mapping ─────────────

def _dp(b: int) -> int:  # 24(C1)–108(C8)
    return 24 + int(b / 255 * 84)

def _dv(b: int) -> int:
    return 40 + ((b >> 1) & 0x3F)

def _dd(b: int) -> int:
    return 120 + popcount(b) * 30

def _dc(i: int) -> int:
    return (i // 4) % 8

def default_mapping(name: str = "default") -> ByteMapping:
    return ByteMapping(name, _dp, _dv, _dd, _dc)

# ───────────── CLI ─────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sonify any binary file → MIDI")
    p.add_argument("binary")
    p.add_argument("out_midi")
    p.add_argument("--tempo", "-t", type=int, default=120)
    p.add_argument("--ticks", "-ppq", type=int, default=480)
    args = p.parse_args()

    Sonifier(args.binary, tempo_bpm=args.tempo, ticks_per_beat=args.ticks) \
        .add_mapping(default_mapping()) \
        .write_midi(args.out_midi)
    print(f"Wrote {args.out_midi}")
