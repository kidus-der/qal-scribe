import re
from pathlib import Path

import pytest

from qal_scribe import cli, models

DOCS = Path(__file__).parents[1] / "docs" / "languages.md"
EXPERIMENTAL = [code for code, spec in models.MODELS.items() if spec.experimental]


def test_specs_are_pinned_and_consistent():
    for code, spec in models.MODELS.items():
        assert spec.language == code
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), code
        assert spec.url == f"https://huggingface.co/{spec.repo}"
        assert spec.license and spec.language_name


def test_amharic_is_the_only_stable_language():
    assert [code for code, spec in models.MODELS.items() if not spec.experimental] == ["amh"]


def test_get_unknown_language_lists_the_available_ones():
    with pytest.raises(ValueError, match="amh"):
        models.get("xx")


def test_models_command_lists_every_language(capsys):
    cli.main(["models"])
    out = capsys.readouterr().out
    for spec in models.MODELS.values():
        assert spec.repo in out and spec.license in out


def test_docs_cover_every_experimental_model():
    text = DOCS.read_text(encoding="utf-8")
    for code in EXPERIMENTAL:
        spec = models.MODELS[code]
        assert spec.url in text and spec.license in text and f"`{code}`" in text, code


@pytest.mark.model
@pytest.mark.parametrize("code", EXPERIMENTAL)
def test_experimental_model_loads_and_handles_silence(code):
    """Run via scripts/gpu_lock.py: downloads and loads each model in turn."""
    import numpy as np

    from qal_scribe.engine import Engine

    engine = Engine(models.get(code), use_lm=False)
    engine.load()
    result = engine.transcribe(np.zeros(16000, dtype="float32"))
    assert isinstance(result.text, str) and "[" not in result.text
