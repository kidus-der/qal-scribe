"""Real-model smoke test. Excluded by default; run: uv run python scripts/gpu_lock.py -- uv run pytest -m model"""
import numpy as np
import pytest

pytestmark = pytest.mark.model


def test_engine_loads_and_handles_silence():
    from qal_scribe import models
    from qal_scribe.engine import Engine

    e = Engine(models.get("amh"), use_lm=False)
    e.load()
    r = e.transcribe(np.zeros(16000, dtype="float32"))
    assert isinstance(r.text, str) and r.compute_seconds < 5
