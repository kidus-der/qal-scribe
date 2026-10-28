"""Registry of speech models per language, plus first-run download."""
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
