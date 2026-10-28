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
