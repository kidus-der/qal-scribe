"""Microphone capture at 16 kHz mono.

Contract: Recorder(on_level).start(); audio = recorder.stop() -> float32 numpy array (may be empty).
on_level(rms) is called from the audio thread for each block; keep it cheap.
"""
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16000


class Recorder:
    def __init__(self, on_level: Callable[[float], None] | None = None):
        import sounddevice as sd

        self._chunks: list[np.ndarray] = []
        self._on_level = on_level
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._callback)

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
