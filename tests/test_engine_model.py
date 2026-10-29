"""Real-model tests. Excluded by default; run: uv run python scripts/gpu_lock.py -- uv run pytest -m model

The two clips are the maintainer's local recordings; the tests skip where they are absent.
"""
from pathlib import Path

import numpy as np
import pytest

from qal_scribe import decode, models
from qal_scribe.engine import SAMPLE_RATE, Engine, pick_device, split_quiet
from qal_scribe.textnorm import edit_distance, normalize

pytestmark = pytest.mark.model

AMHARIC = Path("~/Amharic")
SHORT = AMHARIC / "hohe-asr-amharic/recordings/20261027-163048.wav"
LONG = AMHARIC / "mms-tts-amh/outputs/coffee.wav"  # 42.7 s synthetic speech


def load_clip(path: Path) -> np.ndarray:
    sf = pytest.importorskip("soundfile")
    if not path.exists():
        pytest.skip(f"{path} not available")
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE and audio.ndim == 1
    return audio


def loaded_engine() -> Engine:
    e = Engine(models.get("amh"), use_lm=False)
    e.load()
    return e


def test_engine_loads_and_handles_silence():
    r = loaded_engine().transcribe(np.zeros(SAMPLE_RATE, dtype="float32"))
    assert isinstance(r.text, str) and r.compute_seconds < 5


def test_fp16_copy_is_built_then_used_with_the_same_text(monkeypatch, tmp_path):
    if pick_device() == "cpu":
        pytest.skip("the fp16 copy is only made for GPU runs")
    audio = load_clip(SHORT)
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path)
    try:
        first = loaded_engine().transcribe(audio).text  # from the Hugging Face snapshot, writes the copy
        assert models.fp16_copy(models.get("amh"))
        second = loaded_engine().transcribe(audio).text  # from the copy
        assert first and first == second
    finally:
        models.drop_fp16(models.get("amh"))


def test_long_clip_in_chunks_matches_one_pass():
    audio = load_clip(LONG)
    assert len(split_quiet(audio)) == 2
    e = loaded_engine()
    chunked = normalize(e.transcribe(audio).text).split()
    whole = normalize(decode.greedy(e.logits(audio), e.processor, e.spec)).split()
    assert edit_distance(chunked, whole) <= 0.1 * len(whole)


def test_preview_is_a_greedy_draft_of_the_end():
    audio = load_clip(LONG)
    e = loaded_engine()
    draft = e.preview(audio).text
    assert draft and normalize(draft).split()[-1] in normalize(e.transcribe(audio).text)


def test_sentences_are_cut_at_pauses_with_the_real_tokenizer():
    clip = load_clip(SHORT)
    pause = np.zeros(int(0.8 * SAMPLE_RATE), dtype="float32")
    r = loaded_engine().transcribe(np.concatenate([clip, pause, clip]), sentences=True)
    assert len(r.sentences) >= 2 and " ".join(r.sentences) == r.text
