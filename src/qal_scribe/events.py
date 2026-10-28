"""Events the Controller emits. The TUI and plain output only consume these; they never call the engine.

Contract: events are immutable and cheap. `Level` arrives ~20-30 times a second while recording.
"""
from dataclasses import dataclass
from typing import Literal

Mode = Literal["dictate", "translate"]


@dataclass(frozen=True)
class Loading:
    message: str  # e.g. "downloading model 40%", "loading on mps"


@dataclass(frozen=True)
class Ready:
    device: str  # "mps" | "cpu" | "cuda"
    model: str  # model repo id
    language: str  # e.g. "amh"
    lm: bool  # language-model decoding active
    hotkeys: dict[str, Mode]  # human key name -> mode, e.g. {"Right Option": "dictate"}


@dataclass(frozen=True)
class RecordingStarted:
    mode: Mode


@dataclass(frozen=True)
class Level:
    rms: float  # 0..1 mic level of the latest audio block


@dataclass(frozen=True)
class RecordingStopped:
    mode: Mode
    seconds: float


@dataclass(frozen=True)
class Partial:
    text: str  # live preview while still recording; may change


@dataclass(frozen=True)
class Transcribed:
    mode: Mode
    text: str  # final text that was pasted (after post-processing / translation)
    raw: str  # model output before post-processing
    audio_seconds: float
    compute_seconds: float
    pasted: bool


@dataclass(frozen=True)
class Notice:
    message: str
    level: Literal["info", "warning", "error"] = "info"


Event = Loading | Ready | RecordingStarted | Level | RecordingStopped | Partial | Transcribed | Notice
