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
    "tir": ModelSpec(
        language="tir",
        language_name="Tigrinya",
        repo="badrex/Ethio-ASR-tigrinya",
        revision="16b0de6027bc5b3764e10ea576bc9698cf1844a1",
        license="CC-BY-4.0",
        url="https://huggingface.co/badrex/Ethio-ASR-tigrinya",
        experimental=True,
    ),
    "orm": ModelSpec(
        language="orm",
        language_name="Oromo",
        repo="badrex/Ethio-ASR-oromo",
        revision="9b1074af92478ee9146fb46bc9296a36b4ea6f77",
        license="CC-BY-4.0",
        url="https://huggingface.co/badrex/Ethio-ASR-oromo",
        experimental=True,
    ),
    "som": ModelSpec(
        language="som",
        language_name="Somali",
        repo="IbrahimDayax/xls-r-1b-somali-stt-asr",
        revision="a5608ee9a16abc37e95f6e525605a2e173a71965",
        license="CC-BY-NC-4.0",
        url="https://huggingface.co/IbrahimDayax/xls-r-1b-somali-stt-asr",
        experimental=True,
    ),
    "sid": ModelSpec(
        language="sid",
        language_name="Sidaama",
        repo="badrex/Ethio-ASR-sidaama",
        revision="1364139aa7e24a4839ef8cc29db86054236d445c",
        license="CC-BY-4.0",
        url="https://huggingface.co/badrex/Ethio-ASR-sidaama",
        experimental=True,
    ),
    "wal": ModelSpec(
        language="wal",
        language_name="Wolaytta",
        repo="badrex/Ethio-ASR-wolaytta",
        revision="1347128376ed786d64fe0dcc3fdc5b936cceaa6a",
        license="CC-BY-4.0",
        url="https://huggingface.co/badrex/Ethio-ASR-wolaytta",
        experimental=True,
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
