"""Microphone capture at 16 kHz mono, and the start/stop/error sound cues.

Contract: Recorder(on_level, device=None).start(); audio = recorder.stop() -> float32 numpy array (may be empty).
recorder.snapshot() -> the same array so far, while still recording.
device is a name (substring match), an index, or None for the system default (cfg.mic); ValueError when it cannot
be opened (no match, out of range, output-only). A mic that was unplugged is found again: start() retries once on
a fresh device list, and a clip of only zeros makes the next start() use a fresh one.
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
        if isinstance(device, str) and device.isdigit():
            device = int(device)  # a config file can only say "2"
        self._device = device
        self._chunks: list[np.ndarray] = []
        self._on_level = on_level
        self._stream = self._open()

    def _open(self):
        import sounddevice as sd

        try:
            return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=self._device,
                                  callback=self._callback)
        except sd.PortAudioError as e:  # a number out of range or an output-only device; numbers shift with USB/BT
            which = "the default microphone" if self._device is None else f"microphone {self._device!r}"
            raise ValueError(f"cannot open {which}: {e}; `qal-scribe doctor` lists the inputs") from e

    def _reopen(self) -> None:
        """PortAudio lists the devices once, at import, and a stream keeps the device it was opened on: only a
        fresh list and a new stream reach a mic plugged in since, or the new default after one was unplugged."""
        import sounddevice as sd

        with _play_lock:  # no sound cue may be using PortAudio while it restarts
            self._stream.close()
            sd._terminate()
            sd._initialize()
        self._stream = self._open()

    def _callback(self, data, frames, time_info, status) -> None:
        self._chunks.append(data[:, 0].copy())
        if self._on_level:
            self._on_level(float(np.sqrt(np.mean(data**2))))

    @property
    def active(self) -> bool:
        return self._stream.active

    def start(self) -> None:
        import sounddevice as sd

        self._chunks.clear()
        try:
            self._stream.start()
        except sd.PortAudioError:  # the mic is gone: one more try on a fresh device list
            self._reopen()
            self._stream.start()

    def snapshot(self) -> np.ndarray:
        """Audio captured so far, without stopping."""
        chunks = list(self._chunks)  # the audio thread appends concurrently
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")

    def stop(self) -> np.ndarray:
        self._stream.stop()
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype="float32")
        if not audio.any():  # not even room noise: the mic may have gone away during the clip
            with contextlib.suppress(Exception):  # start() reports a mic that cannot be opened
                self._reopen()
        return audio


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
