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


def test_progress_bar_reports_monotone_bytes_from_the_byte_bar_only():
    seen = []
    Bar = models.progress_bar(lambda done, total: seen.append((done, total)))
    files = Bar(total=3, unit="it", name="huggingface_hub.snapshot_download.thread_map")  # the file-count bar
    files.update(1)
    nbytes = Bar(total=0, unit="B", unit_scale=True, name="huggingface_hub.snapshot_download")
    nbytes.total = 1000  # the hub grows the total as it lists files
    for chunk in (100, 400, 500):
        nbytes.update(chunk)
    assert seen == [(100, 1000), (500, 1000), (1000, 1000)]


def test_progress_bar_is_not_silenced_by_disabled_hub_bars():
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    seen = []
    disable_progress_bars()
    try:
        Bar = models.progress_bar(lambda done, total: seen.append(done))
        Bar(total=10, unit="B", name="huggingface_hub.snapshot_download").update(10)
    finally:
        enable_progress_bars()
    assert seen == [10]


def test_download_passes_the_bar_to_the_hub(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, **kw: calls.update(kw) or str(tmp_path))
    models.download(models.MODELS["amh"], with_lm=True, progress=lambda d, t: None)
    assert calls["tqdm_class"] is not None and "lm/am-5gram.bin" in calls["allow_patterns"]
    models.download(models.MODELS["amh"])
    assert calls["tqdm_class"] is None


def test_missing_bytes_counts_only_files_still_to_download(monkeypatch):
    from types import SimpleNamespace as File

    files = [File(file_size=5, will_download=True), File(file_size=7, will_download=False),
             File(file_size=None, will_download=True)]
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, **kw: files if kw["dry_run"] else "")
    assert models.missing_bytes("a/b", "rev", ["*.json"]) == 5


def test_download_problem_names_hub_errors_and_ignores_the_rest():
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    assert models.download_problem(LocalEntryNotFoundError("Cannot reach the hub\nmore")) == "Cannot reach the hub"
    assert models.download_problem(OfflineModeIsEnabled("offline")) == "offline"
    assert models.download_problem(KeyError("x")) is None
