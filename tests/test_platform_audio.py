import sys
import types

import numpy as np
import pytest

from qal_scribe import audio


class InlineThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


@pytest.fixture(autouse=True)
def synchronous_cues(monkeypatch):
    monkeypatch.setattr(audio.threading, "Thread", InlineThread)


def fake_sounddevice(monkeypatch, **overrides):
    sd = types.SimpleNamespace(played=[], opened=[], **overrides)
    sd.play = overrides.get("play", lambda data, rate: sd.played.append((data, rate)))
    sd.InputStream = lambda **kw: sd.opened.append(kw)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def pitch(samples: np.ndarray) -> float:
    return np.fft.rfftfreq(len(samples), 1 / audio.CUE_RATE)[np.abs(np.fft.rfft(samples)).argmax()]


def halves(kind: str) -> tuple[float, float]:
    tone = audio._tone(kind)
    return pitch(tone[: len(tone) // 2]), pitch(tone[len(tone) // 2 :])


def test_cue_is_played_on_the_default_output(monkeypatch):
    sd = fake_sounddevice(monkeypatch)
    audio.play_cue("start")
    ((data, rate),) = sd.played
    assert rate == audio.CUE_RATE and data.dtype == np.float32 and 0.05 < np.abs(data).max() <= 0.2


def test_cues_rise_fall_and_error_is_low():
    start, stop, error = halves("start"), halves("stop"), halves("error")
    assert start[0] < start[1] and stop[0] > stop[1]
    assert max(error) < min(start + stop)


def test_cues_are_cached_and_end_silently():
    assert audio._tone("stop") is audio._tone("stop")
    tone = audio._tone("stop")
    assert abs(tone[0]) < 1e-3 and abs(tone[-1]) < 1e-3  # no click at either end


def test_no_output_device_is_silent(monkeypatch):
    def no_device(data, rate):
        raise OSError("Error querying device -1")

    fake_sounddevice(monkeypatch, play=no_device)
    audio.play_cue("error")


def test_missing_portaudio_is_silent(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # makes `import sounddevice` raise ImportError
    audio.play_cue("start")


@pytest.mark.parametrize(("device", "expected"), [(None, None), ("USB Mic", "USB Mic"), ("2", 2), (3, 3)])
def test_recorder_passes_the_device(monkeypatch, device, expected):
    sd = fake_sounddevice(monkeypatch)
    audio.Recorder(device=device)
    assert sd.opened[0]["device"] == expected and sd.opened[0]["samplerate"] == audio.SAMPLE_RATE


@pytest.mark.parametrize("device", ["99", None])
def test_a_mic_that_cannot_be_opened_is_a_value_error(monkeypatch, device):
    class PortAudioError(Exception):
        pass

    def refuse(**kw):
        raise PortAudioError("Error querying device 99")

    fake_sounddevice(monkeypatch, PortAudioError=PortAudioError).InputStream = refuse
    with pytest.raises(ValueError, match="cannot open .*Error querying device 99.*qal-scribe doctor"):
        audio.Recorder(device=device)
