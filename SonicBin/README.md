# SonicBin – Binary → MIDI Sonification DSL

## 0 Demo
🎧 [**Demo** on Vimeo](https://vimeo.com/1082456060?share=copy)

> **Turn raw bytes into music.**
>
> SonicBin ships a tiny DSL (Python + CLI) that ingests three binary formats — **FSF**, **LLF**, **TBST** — and emits a Standard MIDI File. One flag or a few lines of code is all it takes.

---

## 1 Installation & Requirements

```bash
python3 -m venv .venv       # optional but recommended
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install mido            # single dependency
```

---

## 2 Quick Starts (CLI Syntax Variants)

| Syntax | Purpose | Example |
|--------|---------|---------|
| **Minimal** | Use defaults, write `<input>.mid`. | `python sonicbin.py tests/one_small_file.llf` |
| **Explicit output** | Name the `.mid` yourself. | `python sonicbin.py tests/one_small_file.llf out/demo.mid` |
| **Custom tempo** | BPM in integer. | `python sonicbin.py tests/test_encryption.fsf --tempo 90` |
| **Custom PPQ** | Ticks‑per‑quarter (resolution). | `python sonicbin.py tests/two_nodes.tbst --ticks 240` |
| **Single‑track (type‑0)** | Collapses all voices into one track, channels intact. | `python sonicbin.py tests/one_small_file.llf --merge` |
| **All options at once** | Full control. | `python sonicbin.py tests/test_encryption.fsf demo.mid -t 140 -ppq 1920 --merge` |

---

## 3 Demo Runner (`run.py`)

Saves three illustrative MIDIs to **`output/`** and showcases every DSL feature:

```bash
python run.py
```

Under the hood it executes:

1. **LLF** → default mapping, merged track.  
2. **FSF** → slower tempo, coarse PPQ, merged track.  
3. **TBST** → two independent `ByteMapping`s (bass + lead) for counterpoint.

Open the resulting files in any DAW to see channels 1‑5 co‑habiting a single track.

---

## 4 Python API Cheatsheet

```python
from sonicbin import Sonifier, default_mapping

# one‑liner: defaults + write MIDI
Sonifier("binary.fsf", merge_tracks=True) \
    .add_mapping(default_mapping())          \
    .write_midi("binary.mid")
```

### 4.1 Custom Mapping Example

```python
from sonicbin import ByteMapping, popcount, slice_filter, Sonifier

# Map even bytes to a bass, odds to a lead line
bass = ByteMapping(
    "bass",
    pitch=lambda b: 36 + (b & 0x1F),        # C2–B3
    velocity=lambda b: 50 + popcount(b),
    duration_ticks=lambda b: 240,
    channel=lambda i, _b: 1,
    byte_filter=slice_filter(slice(None, None, 2)),  # even indices
)

lead = ByteMapping(
    "lead",
    pitch=lambda b: 60 + (b % 12),          # C4–B4
    velocity=lambda _b: 80,
    duration_ticks=lambda b: 120 + (b & 0x0F) * 10,
    channel=lambda idx: (idx // 8) % 4 + 2,          # rotates channels 2‑5
    byte_filter=slice_filter(slice(1, None, 2)),     # odd indices
)

Sonifier("tests/two_nodes.tbst", tempo_bpm=140, merge_tracks=True) \
    .add_mapping(bass) \
    .add_mapping(lead) \
    .write_midi("output/counterpoint.mid")
```

---

## 5 Format Parsers At a Glance

| Suffix | Spec summary | Parser function |
|--------|--------------|-----------------|
| `.fsf` | 32‑byte fixed: payload length (1) + payload (≤30) + key (1). | `_parse_fsf` |
| `.llf` | Linked‑list of small files; clusters = 256 B. | `_parse_llf` |
| `.tbst` | Threaded binary search tree (24 B nodes). | `_parse_tbst` |

All other extensions fall back to **raw‑byte** mode.

---

## 6 Testing & CI Guidance

Add this to `pytest.ini` if you want automatic smoke‑tests in CI:

```ini
[pytest]
addopts = -q
```

Then a minimal test in `tests/test_midis.py`:

```python
from pathlib import Path
import mido

mid = mido.MidiFile("output/one_small_file.mid")
assert mid.type in (0, 1)
assert any(msg.type == "note_on" for track in mid.tracks for msg in track)
assert Path("output/one_small_file.mid").stat().st_size > 0
```

---

## 7 Contributing

PRs for new binary parsers, mappings, or DAW export tips are welcome. Open an issue first if unsure.

---

