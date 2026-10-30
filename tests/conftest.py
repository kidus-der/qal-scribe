import pytest


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """An empty Hugging Face cache; `put` adds a file of a pinned revision to it."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    def put(repo: str, revision: str, filename: str) -> None:
        f = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / filename
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")

    return put


@pytest.fixture(autouse=True)
def private_config_dir(tmp_path, monkeypatch):
    """The TUI now writes config.toml (theme, settings): never the user's own."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


@pytest.fixture(autouse=True)
def offline_update_check(tmp_path, monkeypatch):
    """No test asks the real PyPI or touches the real update cache; tests that need a reply patch urlopen."""
    import urllib.request

    from qal_scribe import update

    def no_network(*_, **__):
        raise OSError("tests are offline")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(update, "cache_dir", lambda: tmp_path / "cache")
