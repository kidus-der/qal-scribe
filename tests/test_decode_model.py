"""Real language-model checks. Excluded by default; run: uv run python scripts/gpu_lock.py -- uv run pytest -m model"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qal_scribe.textnorm import edit_distance, normalize

pytestmark = pytest.mark.model

DEV = Path(__file__).parents[1] / "eval" / "data" / "fleurs-am-dev"


def test_real_lm_loads_and_decodes_silence():
    from qal_scribe import decode, models

    spec = models.get("amh")
    lm = decode.load_lm(models.download(spec, with_lm=True), spec)
    assert lm is not None
    silence = np.zeros((100, 416), dtype="float32")
    silence[:, 408] = 20  # the CTC blank
    assert lm.decode(silence) == ""


def word_errors(refs, hyps):
    return sum(edit_distance(r.split(), normalize(h).split()) for r, h in zip(refs, hyps, strict=True))


@pytest.mark.skipif(not DEV.exists(), reason="run eval/fetch_fleurs.py first")
def test_lm_makes_fewer_word_errors_than_greedy_on_fleurs_dev_sample():
    from qal_scribe import decode, models
    from qal_scribe.engine import Engine

    spec = models.get("amh")
    engine = Engine(spec)
    engine.load()
    assert engine.lm is not None
    wavs = sorted(DEV.glob("*.wav"))[:40]
    refs = [normalize(w.with_suffix(".txt").read_text(encoding="utf-8")) for w in wavs]
    logits = [engine.logits(sf.read(w, dtype="float32")[0]) for w in wavs]
    greedy = word_errors(refs, [decode.greedy(x, engine.processor, spec) for x in logits])
    assert word_errors(refs, [engine.lm.decode(x) for x in logits]) < greedy
