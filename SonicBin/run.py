#!/usr/bin/env python3
"""
run.py – Test for sonicbin.py

Creates output/*.mid from the three course-supplied binaries
using different mappings while forcing single-track (merge) mode.
"""

from pathlib import Path

from sonicbin import (
    Sonifier,
    default_mapping,
    ByteMapping,
    popcount,
    slice_filter,
)

ROOT   = Path(__file__).parent
TESTS  = ROOT / "tests"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    # 1) LLF → default mapping
    llf = TESTS / "one_small_file.llf"
    Sonifier(
        llf,
        tempo_bpm=500,
        merge_tracks=True,
    ).add_mapping(default_mapping()).write_midi(
        OUTDIR / f"{llf.stem}.mid"
    )

    # 2) FSF → slower tempo, coarse PPQ
    fsf = TESTS / "test_encryption.fsf"
    Sonifier(
        fsf,
        tempo_bpm=90,
        ticks_per_beat=240,
        merge_tracks=True,
    ).add_mapping(default_mapping()).write_midi(
        OUTDIR / f"{fsf.stem}.mid"
    )

    # 3) TBST → custom counterpoint demo (bass + lead)
    tbst = TESTS / "two_nodes.tbst"

    def bass_pitch(b: int) -> int:     # C2–B3
        return 36 + (b & 0x1F)

    bass = ByteMapping(
        "bass",
        pitch=bass_pitch,
        velocity=lambda b: 50 + popcount(b),
        duration_ticks=lambda _b: 240,
        channel=lambda _i, _b: 1,
        byte_filter=slice_filter(slice(None, None, 2)),   # even bytes
    )

    def lead_pitch(b: int) -> int:     # C4–B4
        return 60 + (b % 12)

    lead = ByteMapping(
        "lead",
        pitch=lead_pitch,
        velocity=lambda _b: 80,
        duration_ticks=lambda b: 120 + (b & 0x0F) * 10,
        channel=lambda idx: (idx // 8) % 4 + 2,           # cycles 2-5
        byte_filter=slice_filter(slice(1, None, 2)),      # odd bytes
    )

    Sonifier(
        tbst,
        tempo_bpm=140,
        merge_tracks=True,
    ).add_mapping(bass).add_mapping(lead).write_midi(
        OUTDIR / f"{tbst.stem}.mid"
    )

    print("✓  All test MIDIs written to", OUTDIR)


if __name__ == "__main__":
    main()
