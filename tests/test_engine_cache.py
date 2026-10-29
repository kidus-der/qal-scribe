import dataclasses
import sys
import types
from pathlib import Path

import pytest

from qal_scribe import engine as eng
from qal_scribe import models

SPEC = models.get("amh")


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path)
    return tmp_path


def build(folder: Path) -> None:
    (folder / "model.safetensors").write_bytes(b"w" * 100)
    (folder / "config.json").write_text("{}")


def test_cache_dir_per_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(models.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(models.sys, "platform", "darwin")
    assert models.cache_dir() == tmp_path / "Library" / "Caches" / "qal-scribe"
    monkeypatch.setattr(models.sys, "platform", "linux")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert models.cache_dir() == tmp_path / ".cache" / "qal-scribe"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert models.cache_dir() == tmp_path / "xdg" / "qal-scribe"


def test_saving_removes_copies_of_older_revisions_only(cache):
    old = dataclasses.replace(SPEC, revision="0" * 40)
    other_repo = dataclasses.replace(SPEC, repo="someone/else")
    for spec in (old, other_repo):
        models.save_fp16(spec, build)
    in_progress = models.fp16_dir(SPEC).with_name(models.fp16_dir(SPEC).name + ".tmp123")
    in_progress.mkdir()
    models.save_fp16(SPEC, build)
    assert models.fp16_copy(SPEC) and models.fp16_copy(other_repo) and in_progress.exists()
    assert models.fp16_copy(old) is None and not models.fp16_dir(old).exists()


def test_copy_is_keyed_by_repo_and_revision(cache):
    assert models.fp16_dir(SPEC).name == f"snapwre--hohe-asr-amharic@{SPEC.revision}"


def test_saved_copy_is_found(cache):
    assert models.fp16_copy(SPEC) is None
    models.save_fp16(SPEC, build)
    assert models.fp16_copy(SPEC) == models.fp16_dir(SPEC)
    assert [p.name for p in models.fp16_dir(SPEC).parent.iterdir()] == [models.fp16_dir(SPEC).name]  # no temp left


@pytest.mark.parametrize("damage", ["truncate", "delete_weights", "delete_manifest", "garbage_manifest"])
def test_damaged_copy_is_ignored(cache, damage):
    models.save_fp16(SPEC, build)
    folder = models.fp16_dir(SPEC)
    match damage:
        case "truncate":
            (folder / "model.safetensors").write_bytes(b"w" * 40)
        case "delete_weights":
            (folder / "model.safetensors").unlink()
        case "delete_manifest":
            (folder / models.MANIFEST).unlink()
        case "garbage_manifest":
            (folder / models.MANIFEST).write_text("not json")
    assert models.fp16_copy(SPEC) is None


def test_failed_build_leaves_nothing_and_keeps_the_old_copy(cache):
    models.save_fp16(SPEC, build)

    def broken(folder):
        (folder / "model.safetensors").write_bytes(b"half")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        models.save_fp16(SPEC, broken)
    assert models.fp16_copy(SPEC) is not None
    assert len(list(models.fp16_dir(SPEC).parent.iterdir())) == 1


def install_fake_transformers(monkeypatch, fail_on: Path | None):
    loaded = []

    class Part:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, folder):
            (folder / f"{self.name}.json").write_text("{}")
            if self.name == "model":
                (folder / "model.safetensors").write_bytes(b"fp16" * 10)

        def eval(self):
            return self

    def auto(name):
        def from_pretrained(path, **_):
            loaded.append((name, Path(path)))
            if fail_on and Path(path) == fail_on:
                raise OSError("corrupt safetensors")
            return Part(name)

        return types.SimpleNamespace(from_pretrained=from_pretrained)

    fake = types.SimpleNamespace(AutoModelForCTC=auto("model"), AutoProcessor=auto("processor"))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return loaded


def gpu_engine():
    torch = pytest.importorskip("torch")
    e = eng.Engine(SPEC, device="mps")
    e.dtype = torch.float16
    return e


def test_first_gpu_load_builds_the_copy_and_later_loads_use_it(cache, monkeypatch, tmp_path):
    loaded = install_fake_transformers(monkeypatch, fail_on=None)
    e, msgs = gpu_engine(), []
    e._load_weights(tmp_path / "snapshot", msgs.append)
    assert {p for _, p in loaded} == {tmp_path / "snapshot"} and models.fp16_copy(SPEC)
    loaded.clear()
    msgs.clear()
    e._load_weights(tmp_path / "snapshot", msgs.append)
    assert {p for _, p in loaded} == {models.fp16_dir(SPEC)} and not msgs


def test_corrupt_copy_falls_back_to_the_snapshot_and_is_rebuilt(cache, monkeypatch, tmp_path):
    models.save_fp16(SPEC, build)
    loaded = install_fake_transformers(monkeypatch, fail_on=models.fp16_dir(SPEC))
    e, msgs = gpu_engine(), []
    processor, model = e._load_weights(tmp_path / "snapshot", msgs.append)
    assert (processor.name, model.name) == ("processor", "model")
    assert loaded[-1][1] == tmp_path / "snapshot"
    assert any("rebuilding" in m for m in msgs)
    assert models.fp16_copy(SPEC) and (models.fp16_dir(SPEC) / "model.json").exists()  # rebuilt from the snapshot


def test_cpu_does_not_use_or_write_the_copy(cache, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    loaded = install_fake_transformers(monkeypatch, fail_on=None)
    e = eng.Engine(SPEC, device="cpu")
    e.dtype = torch.float32
    e._load_weights(tmp_path / "snapshot", print)
    assert {p for _, p in loaded} == {tmp_path / "snapshot"} and models.fp16_copy(SPEC) is None


def test_unwritable_cache_does_not_stop_loading(cache, monkeypatch, tmp_path):
    install_fake_transformers(monkeypatch, fail_on=None)
    monkeypatch.setattr(models, "save_fp16", lambda *_: (_ for _ in ()).throw(OSError("read-only")))
    msgs = []
    _, model = gpu_engine()._load_weights(tmp_path / "snapshot", msgs.append)
    assert model.name == "model" and any("read-only" in m for m in msgs)
