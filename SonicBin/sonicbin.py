#!/usr/bin/env python3
"""
sonicbin.py – Hear your data (FSF / LLF / TBST aware)

Binary-to-MIDI sonification toolkit for CS 343S “Binary Clinic”.

•   Import as a library (see run.py) or run directly:
        $ python sonicbin.py tests/one_small_file.llf --merge

•   When --merge is given, the file is written as SMF type-0
    (one track) but each part keeps its own MIDI channel so
    counterpoint survives the merge.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

try:
    from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install mido:  pip install mido") from exc

__all__ = [
    "ByteMapping",
    "Sonifier",
    "default_mapping",
    "popcount",
    "slice_filter",
]

# ────────────────────────────────────────────── helpers
PopFunc     = Callable[[int], int]
ChannelFunc = Callable[[int], int] | Callable[[int, int], int]


def popcount(x: int) -> int:
    return bin(x & 0xFF).count("1")


def slice_filter(slc: slice) -> Callable[[int], bool]:
    start, stop, step = slc.start or 0, slc.stop, slc.step or 1
    if stop is None:
        return lambda i: i >= start and (i - start) % step == 0
    return lambda i: start <= i < stop and (i - start) % step == 0

# ────────────────────────────────────────────── parsers
def _parse_fsf(raw: bytes) -> bytes:
    if len(raw) != 32:
        raise ValueError(".fsf file must be exactly 32 bytes")
    length    = raw[0]
    encrypted = raw[1:31][:length]
    key       = raw[-1]
    return bytes(((b - key) & 0xFF) for b in encrypted)


def _parse_llf(raw: bytes) -> bytes:
    if len(raw) < 8:
        raise ValueError(".llf file too small")
    n_files, first_desc = struct.unpack_from("<II", raw, 0)
    CLUSTER, base = 256, 8  # clusters start right after header

    def cluster(idx: int) -> bytes:
        if not idx:
            return b""
        off = base + (idx - 1) * CLUSTER
        return raw[off:off + CLUSTER]

    payload  = bytearray()
    desc_idx = first_desc
    for _ in range(n_files):
        if not desc_idx:
            break
        c             = cluster(desc_idx)
        next_link, sz = struct.unpack_from("<IB", c, 0)
        first_content = struct.unpack_from("<I", c, 5)[0]

        cont_idx = first_content
        while cont_idx:
            cc = cluster(cont_idx)
            nxt, cnt = struct.unpack_from("<IB", cc, 0)
            payload.extend(cc[5:5 + cnt])
            cont_idx = nxt
        desc_idx = next_link
    return bytes(payload)


def _parse_tbst(raw: bytes) -> bytes:
    if len(raw) < 8:
        raise ValueError(".tbst file too small")
    root_ptr, = struct.unpack_from("<I", raw, 0)
    NODE = 24
    visited, out = set(), bytearray()

    def node(ptr: int) -> bytes:
        return raw[ptr:ptr + NODE]

    def traverse(ptr: int):
        if ptr == 0 or ptr in visited or ptr + NODE > len(raw):
            return
        visited.add(ptr)
        n               = node(ptr)
        key, val, *_pad = struct.unpack_from("<IIIIII", n, 0)
        left, right     = _pad[-2:]
        traverse(left)
        out.extend(struct.pack("<II", key, val))
        traverse(right)

    traverse(root_ptr)
    return bytes(out)

# ────────────────────────────────────────────── core
class ByteMapping:
    """
    Maps selected bytes to (pitch, velocity, duration, channel).
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
    ):
        self.name    = name
        self.pitch_f = pitch
        self.vel_f   = velocity
        self.dur_f   = duration_ticks
        self.chan_f  = channel
        self.byte_ok = byte_filter or (lambda _i: True)

    def map(self, idx: int, b: int) -> Optional[Tuple[int, int, int, int]]:
        if not self.byte_ok(idx):
            return None
        pitch = max(0, min(127, self.pitch_f(b)))
        veloc = max(1, min(127, self.vel_f(b)))
        dur   = max(1, self.dur_f(b))
        try:
            chan = self.chan_f(idx, b)
        except TypeError:
            chan = self.chan_f(idx)
        return pitch, veloc, dur, chan % 16


class Sonifier:
    """
    Facade: load → add mapping(s) → write .mid
    """

    _PARSERS = {
        ".fsf": _parse_fsf,
        ".llf": _parse_llf,
        ".tbst": _parse_tbst,
    }

    def __init__(
        self,
        binary_path: Union[str, Path],
        *,
        tempo_bpm: int = 120,
        ticks_per_beat: int = 480,
        merge_tracks: bool = False,
    ):
        self.path = Path(binary_path)
        if not self.path.is_file():
            raise FileNotFoundError(binary_path)
        raw          = self.path.read_bytes()
        parser       = self._PARSERS.get(self.path.suffix.lower())
        self.data    = parser(raw) if parser else raw
        self.tempo   = tempo_bpm
        self.ppq     = ticks_per_beat
        self.merge   = merge_tracks
        self.mapping: List[ByteMapping] = []

    def add_mapping(self, m: ByteMapping) -> "Sonifier":
        self.mapping.append(m)
        return self

    def write_midi(self, out_path: Union[str, Path]) -> Path:
        if not self.mapping:
            raise RuntimeError("Add at least one ByteMapping before writing MIDI")

        mf   = MidiFile(type=0 if self.merge else 1, ticks_per_beat=self.ppq)
        meta = MidiTrack()
        mf.tracks.append(meta)
        meta.append(MetaMessage("set_tempo", tempo=bpm2tempo(self.tempo), time=0))

        # One logical “track array” so code below can still address tracks[ch]
        tracks = [meta] * 16 if self.merge else [MidiTrack() for _ in range(16)]
        if not self.merge:
            mf.tracks.extend(tracks)

        for i, b in enumerate(self.data):
            for m in self.mapping:
                mapped = m.map(i, b)
                if mapped is None:
                    continue
                p, v, d, ch = mapped
                tr = tracks[ch]            # keep original channel even if merged
                tr.append(Message("note_on",  channel=ch, note=p, velocity=v, time=0))
                tr.append(Message("note_off", channel=ch, note=p, velocity=0, time=d))

        mf.save(str(out_path))
        return Path(out_path).resolve()

# ────────────────────────────────────────────── default mapping
def _d_pitch(b: int)    -> int: return 24 + int(b / 255 * 84)   # C1–C8
def _d_velocity(b: int) -> int: return 40 + ((b >> 1) & 0x3F)
def _d_duration(b: int) -> int: return 120 + popcount(b) * 30
def _d_channel(i: int)  -> int: return (i // 4) % 8


def default_mapping(name: str = "default") -> ByteMapping:
    return ByteMapping(name, _d_pitch, _d_velocity, _d_duration, _d_channel)

# ────────────────────────────────────────────── CLI
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Binary → Standard MIDI File")
    ap.add_argument("binary")
    ap.add_argument("out_midi", nargs="?", help="Defaults to <input>.mid")
    ap.add_argument("--tempo", "-t",   type=int, default=120, help="Tempo (BPM)")
    ap.add_argument("--ticks", "-ppq", type=int, default=480, help="PPQ resolution")
    ap.add_argument("--merge", action="store_true", help="Single-track output")
    args = ap.parse_args()

    out_name = args.out_midi or f"{Path(args.binary).with_suffix('').name}.mid"
    try:
        Sonifier(
            args.binary,
            tempo_bpm=args.tempo,
            ticks_per_beat=args.ticks,
            merge_tracks=args.merge,
        ).add_mapping(default_mapping()).write_midi(out_name)
        print(f"✓  Wrote {out_name}")
    except Exception as exc:
        print(f"✗  {exc}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
