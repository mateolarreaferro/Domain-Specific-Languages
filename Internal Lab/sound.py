#!/usr/bin/env python3
from typing import Self, BinaryIO
import subprocess as sub
from tempfile import NamedTemporaryFile
from dataclasses import dataclass


@dataclass
class Volume:
    """The volume for a sound; wraps a non‑negative float"""
    volume: float

    def __post_init__(self):
        if self.volume < 0:
            raise ValueError(f"Volume must be in range [0.0, ); got {self.volume}")


@dataclass
class Speed:
    """The speed for a sound; wraps a positive float"""
    speedup: float

    def __post_init__(self):
        if self.speedup <= 0:
            raise ValueError(f"Speedup must be in range (0.0, ); got {self.speedup}")


@dataclass
class Sound:
    """A wrapper around an audio file that supports sound algebra."""
    file: str | BinaryIO

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def path(self) -> str:
        """Return the on‑disk path for the underlying file."""
        return self.file if isinstance(self.file, str) else self.file.name

    def _temp_mp3(self) -> "Sound":
        """Create a scratch mp3 file wrapped as a Sound."""
        return Sound(NamedTemporaryFile(suffix=".mp3"))


    # Convenience
    def play(self):
        """Play this sound (blocking)."""
        sub.run(
            f"ffplay -loglevel quiet -nodisp -autoexit {self.path()}",
            shell=True,
            check=True,
        )


    # Operator overloading
    def __or__(self, other: Self) -> Self:
        """Sequence two sounds (self followed by other)."""
        out = self._temp_mp3()
        sub.run(f"cat {self.path()} {other.path()} > {out.path()}", shell=True, check=True)
        return out

    def __and__(self, other: Self) -> Self:
        """Overlay two sounds so they start together."""
        out = self._temp_mp3()
        cmd = (
            f"ffmpeg -loglevel quiet -y -i {self.path()} -i {other.path()} "
            f"-filter_complex amix=inputs=2:duration=longest {out.path()}"
        )
        sub.run(cmd, shell=True, check=True)
        return out

    def __mul__(self, other: int) -> Self:
        """Repeat this sound *other* times (x * 3)."""
        if not isinstance(other, int):
            return NotImplemented
        if other < 1:
            raise ValueError("Repeat count must be a positive integer")
        out = self._temp_mp3()
        cat_args = " ".join([self.path()] * other)
        sub.run(f"cat {cat_args} > {out.path()}", shell=True, check=True)
        return out

    def __rmul__(self, other: int) -> Self:
        """Repeat this sound *other* times (3 * x)."""
        return self.__mul__(other)

    def __matmul__(self, other: Volume | Speed) -> Self:
        """Apply a Volume or Speed modification."""
        out = self._temp_mp3()

        # Volume change
        if isinstance(other, Volume):
            cmd = (
                f"ffmpeg -loglevel quiet -y -i {self.path()} "
                f"-filter:a \"volume={other.volume}\" {out.path()}"
            )
            sub.run(cmd, shell=True, check=True)
            return out

        # Speed change
        if isinstance(other, Speed):
            s = other.speedup
            if s < 0.5:
                raise ValueError("Speedup below 0.5 not supported by atempo")
            # chain atempo so every factor is within [0.5,2]
            factors = []
            while s > 2.0:
                factors.append(2.0)
                s /= 2.0
            factors.append(s)
            filt = ",".join(f"atempo={f}" for f in factors)
            cmd = (
                f"ffmpeg -loglevel quiet -y -i {self.path()} "
                f"-filter:a \"{filt}\" {out.path()}"
            )
            sub.run(cmd, shell=True, check=True)
            return out

        raise TypeError("@ expects Volume or Speed")

    def __getitem__(self, slice_: slice) -> Self:
        """Slice a sound with indices in seconds."""
        if slice_.step is not None:
            raise ValueError("Step not supported in sound slicing")

        start = slice_.start
        end = slice_.stop
        if (start is not None and start < 0) or (end is not None and end < 0):
            raise ValueError("Negative indices not supported")

        # no slice → return original
        if start is None and end is None:
            return self

        out = self._temp_mp3()
        # Build atrim filter string
        filt_parts = []
        if start is not None:
            filt_parts.append(f"start={start}")
        if end is not None:
            filt_parts.append(f"end={end}")
        filt = "atrim=" + ":".join(filt_parts) + ",asetpts=PTS-STARTPTS"

        cmd = (
            f"ffmpeg -loglevel quiet -y -i {self.path()} "
            f"-af \"{filt}\" {out.path()}"
        )
        sub.run(cmd, shell=True, check=True)
        return out
