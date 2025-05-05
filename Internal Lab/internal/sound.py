#!/usr/bin/env python3


from typing import Self, BinaryIO
import subprocess as sub
from tempfile import NamedTemporaryFile
from dataclasses import dataclass


@dataclass
class Volume:
    """The volume for a sound; wraps a non-negative float"""

    volume: float

    def __post_init__(self):
        if self.volume < 0:
            raise ValueError(f"Volume must be in range [0.0, ]; got {self.volume}")


@dataclass
class Speed:
    """The speed for a sound; wraps a positive float"""

    speedup: float

    def __post_init__(self):
        if self.speedup <= 0:
            raise ValueError(f"Speedup must be in range (0.0, ]; got {self.speedup}")


@dataclass
class Sound:
    """The speed for a sound; wraps a positive float"""

    file: str | BinaryIO

    def path(self) -> str:
        """Create a sound from a path to an audio file (e.g., a *.mp3)"""
        if isinstance(self.file, str):
            return self.file
        else:
            return self.file.name

    def play(self):
        """Play this sound, blocking until complete."""
        sub.run(
            f"ffplay -loglevel quiet -nodisp -autoexit {self.path()}",
            shell=True,
            check=True,
        )

    def __or__(self, other: Self) -> Self:
        """The sound that plays self, then other.

        See: unix's cat (really!)
        """
        out = Sound(NamedTemporaryFile(suffix=".mp3"))
        sub.run(
            f"cat {self.path()} {other.path()} > {out.path()}", shell=True, check=True
        )
        return out

    def __and__(self, other: Self) -> Self:
        """The sound that plays self and other simultaneously.
        They start at the same time, but one might end before the other.

        See: https://ffmpeg.org/ffmpeg-filters.html#amix
        """
        pass  # TODO (we have ~7 lines)

    def __mul__(self, other: int) -> Self:
        """Repeat this sound 'other' times.

        See: unix's cat (really!)
        """
        pass  # TODO (we have ~11 lines)

    def __rmul__(self, other: int) -> Self:
        """Repeat this sound 'other' times."""
        pass  # TODO (we have ~4 lines)

    def __matmul__(self, other: Volume | Speed) -> Self:
        """Modify the volume or speed of a sound."""
        pass  # TODO (we have ~18 lines)

    def __getitem__(self, slice_: slice) -> Self:
        """Slice a sound, with indices in seconds.

        See: https://ffmpeg.org/ffmpeg-filters.html#atrim
        """
        pass  # TODO (we have ~16 lines)
