import numpy as np
import pytest

from qal_scribe import engine as eng
from qal_scribe.engine import MAX_CHUNK, SAMPLE_RATE, split_quiet

rng = np.random.default_rng(0)


def speech(seconds: float, pause: float = 0.4) -> np.ndarray:
    """Noise 'words' of 0.5-1.2 s separated by exact-silence pauses."""
    parts, total = [], 0.0
    while total < seconds:
        word = rng.uniform(0.5, 1.2)
        parts += [rng.uniform(-0.3, 0.3, int(word * SAMPLE_RATE)), np.zeros(int(pause * SAMPLE_RATE))]
        total += word + pause
    return np.concatenate(parts)[: int(seconds * SAMPLE_RATE)].astype("float32")


def test_short_audio_is_one_chunk():
    audio = speech(20)
    (chunk,) = split_quiet(audio)
    assert chunk is audio
    assert len(split_quiet(np.ones(MAX_CHUNK, dtype="float32"))) == 1


@pytest.mark.parametrize("seconds", [42.7, 61, 95, 180])
def test_cuts_fall_inside_pauses(seconds):
    audio = speech(seconds)
    chunks = split_quiet(audio)
    assert np.array_equal(np.concatenate(chunks), audio)  # nothing lost or duplicated
    assert all(len(c) <= MAX_CHUNK for c in chunks)
    assert min(len(c) for c in chunks) > 10 * SAMPLE_RATE
    for cut in np.cumsum([len(c) for c in chunks[:-1]]):
        assert not audio[cut - 800: cut + 800].any(), "cut is mid-word"


@pytest.mark.parametrize(("seconds", "count"), [(42.7, 2), (61, 3)])
def test_uses_the_fewest_chunks_when_cuts_are_free(seconds, count):
    assert len(split_quiet(speech(seconds))) == count


def test_prefers_the_quietest_point():
    audio = rng.uniform(-0.3, 0.3, 40 * SAMPLE_RATE).astype("float32")
    audio[22 * SAMPLE_RATE: int(22.5 * SAMPLE_RATE)] *= 0.01  # one faint pause, others are none
    first, _ = split_quiet(audio)
    assert abs(len(first) / SAMPLE_RATE - 22.25) < 0.2


@pytest.mark.parametrize("extra", [1, SAMPLE_RATE // 100])
def test_just_over_the_limit_is_balanced_not_a_stub(extra):
    audio = rng.uniform(-0.3, 0.3, MAX_CHUNK + extra).astype("float32")
    chunks = split_quiet(audio)
    assert len(chunks) == 2 and all(10 * SAMPLE_RATE < len(c) <= MAX_CHUNK for c in chunks)


def test_exact_multiple_of_the_limit_still_cuts_where_it_is_quiet():
    audio = speech(60)
    audio = np.concatenate([audio, np.zeros(2 * MAX_CHUNK - len(audio), dtype="float32")])
    chunks = split_quiet(audio)
    assert all(len(c) <= MAX_CHUNK for c in chunks) and len(chunks) == 3
    assert np.array_equal(np.concatenate(chunks), audio)


def test_continuous_signal_without_pauses_still_splits():
    audio = np.sin(np.arange(75 * SAMPLE_RATE) * 0.05).astype("float32")
    chunks = split_quiet(audio)
    assert np.array_equal(np.concatenate(chunks), audio) and all(len(c) <= MAX_CHUNK for c in chunks)


def test_transcribe_joins_chunk_texts(monkeypatch):
    monkeypatch.setattr(eng.Engine, "logits", lambda self, audio: audio)
    monkeypatch.setattr(eng.decode, "decode", lambda logits, *_: "" if len(logits) < 5 * SAMPLE_RATE else "ሰላም")
    e = eng.Engine(eng.models.get("amh"))
    long = speech(70)
    assert e.transcribe(long).text == "ሰላም ሰላም ሰላም"
    assert e.transcribe(long[: SAMPLE_RATE]).text == ""
