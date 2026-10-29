"""Real-model check. Excluded by default; run: uv run python scripts/gpu_lock.py -- uv run pytest -m model"""
import pytest

pytestmark = pytest.mark.model


def test_translates_a_known_sentence():
    from qal_scribe.engine import pick_device
    from qal_scribe.translate import Translator

    t = Translator(pick_device())
    t.load()
    out = t.translate("ካዛብላንካ በሁሉም ሞሮኮ ውስጥ ለመገብየት በዝቅተኛ ደረጃ አስደሳች ከሆኑ ቦታዎች አንዱ ነው። ")
    assert "Casablanca" in out and "Morocco" in out
    assert t.translate("") == ""
