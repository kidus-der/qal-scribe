"""Registry of speech models per language, plus first-run download and the fp16 weight cache."""
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    language: str  # ISO 639-3
    language_name: str
    repo: str
    revision: str  # pinned commit for reproducibility
    license: str
    url: str
    lm_file: str | None = None  # path inside the repo
    output_tag: str | None = None  # tag the model prefixes to its output, stripped
    experimental: bool = False


MODELS: dict[str, ModelSpec] = {
    "amh": ModelSpec(
        language="amh",
        language_name="Amharic",
        repo="snapwre/hohe-asr-amharic",
        revision="217712df448adc0956b2d1b3f13bc30466c8733b",
        license="CC-BY-4.0",
        url="https://huggingface.co/snapwre/hohe-asr-amharic",
        lm_file="lm/am-5gram.bin",
        output_tag="[AMH]",
    ),
}


def get(language: str) -> ModelSpec:
    try:
        return MODELS[language]
    except KeyError:
        raise ValueError(f"no model for language {language!r}; available: {', '.join(MODELS)}") from None


def download(spec: ModelSpec, with_lm: bool = False) -> Path:
    """Fetch the model into the Hugging Face cache (no-op when present) and return the local folder."""
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors"] + ([spec.lm_file] if with_lm and spec.lm_file else [])
    return Path(snapshot_download(spec.repo, revision=spec.revision, allow_patterns=patterns))


MANIFEST = "manifest.json"  # written last: file name -> size, so a half-written or damaged copy is never trusted


def cache_dir() -> Path:
    """macOS ~/Library/Caches/qal-scribe; elsewhere $XDG_CACHE_HOME/qal-scribe (default ~/.cache)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "qal-scribe"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "qal-scribe"


def fp16_dir(spec: ModelSpec) -> Path:
    return cache_dir() / "fp16" / f"{spec.repo.replace('/', '--')}@{spec.revision}"


def _sizes(folder: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in sorted(folder.iterdir()) if p.name != MANIFEST}


def fp16_copy(spec: ModelSpec) -> Path | None:
    """The finished fp16 copy of the model (weights + config/processor files), or None when absent or damaged."""
    folder = fp16_dir(spec)
    try:
        return folder if json.loads((folder / MANIFEST).read_text()) == _sizes(folder) else None
    except (OSError, ValueError):
        return None


def save_fp16(spec: ModelSpec, write: Callable[[Path], None]) -> None:
    """Build the fp16 copy: `write(folder)` fills a temp folder that is renamed into place only when complete.
    Raises OSError (disk full, read-only cache); callers treat the cache as optional."""
    final = fp16_dir(spec)
    tmp = final.with_name(f"{final.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        write(tmp)
        (tmp / MANIFEST).write_text(json.dumps(_sizes(tmp)))
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        prefix = final.name.split("@")[0] + "@"
        for stale in final.parent.glob(f"{prefix}*"):  # copies of older pinned revisions are dead weight (1.2 GB)
            if stale != final and ".tmp" not in stale.name:
                shutil.rmtree(stale, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def drop_fp16(spec: ModelSpec) -> None:
    shutil.rmtree(fp16_dir(spec), ignore_errors=True)
