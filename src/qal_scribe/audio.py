"""Microphone capture at 16 kHz mono, and the start/stop/error sound cues.

Contract: Recorder(on_level, device=None).start(); audio = recorder.stop() -> float32 numpy array (may be empty).
device is a name (substring match), an index, or None for the system default (cfg.mic).
on_level(rms) is called from the audio thread for each block; keep it cheap.
play_cue(kind) never blocks, never raises, and does nothing when there is no output device.
"""
import contextlib
import functools
import threading
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16000
CUE_RATE = 48000
CUE_GAIN = 0.15  # quiet: a cue, not an alert
NOTE_SECONDS = 0.08
CUES = {"start": (660, 880), "stop": (880, 660), "error": (220, 165)}  # Hz: rising, falling, low


class Recorder:
    def __init__(self, on_level: Callable[[float], None] | None = None, device: str | int | None = None):
        import sounddevice as sd

        if isinstance(device, str) and device.isdigit():
            device = int(device)  # a config file can only say "2"
        self._chunks: list[np.ndarray] = []
        self._on_level = on_level
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device,
                                      callback=self._callback)

    def _callback(self, data, frames, time_info, status) -> None:
        self._chunks.append(data[:, 0].copy())
        if self._on_level:
            self._on_level(float(np.sqrt(np.mean(data**2))))

    @property
    def active(self) -> bool:
        return self._stream.active

    def start(self) -> None:
        self._chunks.clear()
        self._stream.start()

    def stop(self) -> np.ndarray:
        self._stream.stop()
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype="float32")


@functools.cache
def _tone(kind: str) -> np.ndarray:
    n = int(CUE_RATE * NOTE_SECONDS)
    t = np.arange(n) / CUE_RATE
    notes = [np.sin(2 * np.pi * f * t) * np.hanning(n) for f in CUES[kind]]  # hanning fades in/out: no click
    return (np.concatenate(notes) * CUE_GAIN).astype("float32")


_play_lock = threading.Lock()  # sd.play keeps one global stream; two cues must not race on it


def _play(data: np.ndarray) -> None:
    # sd.play uses its own output stream, so it never touches the Recorder's input stream.
    with contextlib.suppress(Exception), _play_lock:  # cues are cosmetic: no output device, no PortAudio, busy
        import sounddevice as sd

        sd.play(data, CUE_RATE)


def play_cue(kind: str) -> None:
    """Short sound when recording starts/stops ("start" | "stop" | "error"). Must not block. Workstream `platform`."""
    # Opening the output stream takes 40-110 ms on macOS, too long for pynput's callback thread, hence the thread.
    threading.Thread(target=_play, args=(_tone(kind),), daemon=True).start()  # unknown kind: KeyError, a bug
