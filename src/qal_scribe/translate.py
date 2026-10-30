"""Amharic -> English with a model of the user's choice (TRANSLATION_MODELS; `translation_model` in the config).
Loaded lazily on first use.

Contract: t = Translator(device, model="nllb-600m"); t.load(); t.translate(amharic_text, language="amh") -> english_text
download(model) -> folder fetches the weights (what load() asks for).
`amharic_text` is a string or a list of sentences (the Controller cuts speech at pauses). Empty or whitespace-only
text gives "" without loading anything. Long text is split into sentences (and, for text without punctuation, into
chunks of MAX_WORDS words) and translated as one batch. `language` is a qal-scribe language code; one NLLB does not
know (Sidaama, Wolaytta) raises ValueError.
Licence: the NLLB models are CC-BY-NC-4.0 (non-commercial), MADLAD is Apache-2.0; see NOTICE.
"""
import re
from dataclasses import dataclass

NLLB_CODES = {"amh": "amh_Ethi", "tir": "tir_Ethi", "orm": "gaz_Latn", "som": "som_Latn"}
SRC_LANG = NLLB_CODES["amh"]
TGT_LANG = "eng_Latn"


@dataclass(frozen=True)
class TranslationSpec:
    repo: str
    revision: str  # pinned commit for reproducibility
    license: str
    weights: str  # the file whose presence in the cache means "downloaded"
    patterns: tuple[str, ...]  # what to fetch: the repos also hold GGUF and duplicate weights we never read
    size_gb: float  # download size, for messages
    half: str  # torch dtype name used on mps/cuda
    prefix: str = ""  # MADLAD takes the target language as a "<2en> " tag on the input; NLLB forces a start token


TRANSLATION_MODELS: dict[str, TranslationSpec] = {
    "nllb-600m": TranslationSpec("facebook/nllb-200-distilled-600M", "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
                                 "CC-BY-NC-4.0", "pytorch_model.bin", ("*.json", "*.bin", "*.model"), 2.5, "float16"),
    "nllb-1.3b": TranslationSpec("facebook/nllb-200-distilled-1.3B", "7be3e24664b38ce1cac29b8aeed6911aa0cf0576",
                                 "CC-BY-NC-4.0", "pytorch_model.bin", ("*.json", "*.bin", "*.model"), 5.5, "float16"),
    "madlad-3b": TranslationSpec("google/madlad400-3b-mt", "fa184c675da0b5c9e1c8694fccd4e12e2d422094", "Apache-2.0",
                                 "model.safetensors", ("*.json", "*.safetensors", "spiece.model"), 11.8,
                                 "bfloat16", "<2en> "),  # T5 overflows in float16
}
DEFAULT_MODEL = "nllb-600m"


def download(model: str = DEFAULT_MODEL) -> str:
    from huggingface_hub import snapshot_download

    spec = TRANSLATION_MODELS[model]
    return snapshot_download(spec.repo, revision=spec.revision, allow_patterns=list(spec.patterns))


# NLLB is trained on sentences; longer unpunctuated runs (raw ASR output) degrade or truncate.
# Only the fallback for a stretch of speech without a pause: the Controller already cuts at pauses.
MAX_WORDS = 40
# Ethiopic full stop and question mark end a sentence even without a following space.
_SENTENCE_END = re.compile(r"(?<=[።፧])\s*|(?<=[.!?])\s+")


def split_sentences(text: str | list[str]) -> list[str]:
    chunks = []
    for sentence in (s for t in ([text] if isinstance(text, str) else text) for s in _SENTENCE_END.split(t)):
        words = sentence.split()
        chunks += [" ".join(words[i : i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)]
    return chunks


class Translator:
    def __init__(self, device: str = "cpu", dtype=None, model: str = DEFAULT_MODEL):
        """dtype: a torch dtype; default is the spec's half precision on mps/cuda (measured equal to fp32 for
        NLLB by eval/translate_eval.py), float32 on the CPU."""
        self.name, self.spec = model, TRANSLATION_MODELS[model]
        self.device = device
        self.dtype = dtype
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = download(self.name)
        if self.spec.prefix:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, src_lang=SRC_LANG)
            self.target_id = self.tokenizer.convert_tokens_to_ids(TGT_LANG)
        dtype = self.dtype or (getattr(torch, self.spec.half) if self.device in ("mps", "cuda") else torch.float32)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).eval()
        try:
            self.model = model.to(self.device)
        except (RuntimeError, AssertionError):  # device unavailable or lacks an op: fall back to the CPU
            self.device = "cpu"
            self.model = model.float().to("cpu")

    def translate(self, text: str | list[str], language: str = "amh") -> str:
        if language not in NLLB_CODES:
            raise ValueError(f"no translation model for language {language!r}; translation knows "
                             f"{', '.join(NLLB_CODES)}")
        sentences = split_sentences(text)
        if not sentences:
            return ""
        self.load()
        import torch

        if self.spec.prefix:
            sentences = [self.spec.prefix + s for s in sentences]
            target = {}
        else:
            self.tokenizer.src_lang = NLLB_CODES[language]
            target = {"forced_bos_token_id": self.target_id}
        batch = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=256)
        batch = batch.to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **batch,
                **target,
                num_beams=4,
                no_repeat_ngram_size=4,  # ends the loops NLLB falls into on spoken numbers; chrF unchanged
                max_length=None,  # the model's default of 200 would otherwise conflict with max_new_tokens
                max_new_tokens=2 * batch["input_ids"].shape[1] + 16,  # English is longer than Amharic, not 2x
            )
        # Amharic has no case, so the model often starts sentences in lower case.
        sentences = (s[:1].upper() + s[1:] for s in self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return " ".join(sentences)
