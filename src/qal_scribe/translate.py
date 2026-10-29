"""Amharic -> English with facebook/nllb-200-distilled-600M (workstream `translate`). Loaded lazily on first use.

Contract: t = Translator(device); t.load(); t.translate(amharic_text, language="amh") -> english_text
`amharic_text` is a string or a list of sentences (the Controller cuts speech at pauses). Empty or whitespace-only
text gives "" without loading anything. Long text is split into sentences (and, for text without punctuation, into
chunks of MAX_WORDS words) and translated as one batch. `language` is a qal-scribe language code; one NLLB does not
know (Sidaama, Wolaytta) raises ValueError.
Licence: the model is CC-BY-NC-4.0 (non-commercial); see NOTICE.
"""
import re

NLLB_CODES = {"amh": "amh_Ethi", "tir": "tir_Ethi", "orm": "gaz_Latn", "som": "som_Latn"}
SRC_LANG = NLLB_CODES["amh"]
TGT_LANG = "eng_Latn"
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
    MODEL = "facebook/nllb-200-distilled-600M"
    REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

    def __init__(self, device: str = "cpu", dtype=None):
        """dtype: a torch dtype; default is float16 on mps/cuda (measured equal to fp32 by eval/translate_eval.py)."""
        self.device = device
        self.dtype = dtype
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = snapshot_download(self.MODEL, revision=self.REVISION, allow_patterns=["*.json", "*.bin", "*.model"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, src_lang=SRC_LANG)
        self.target_id = self.tokenizer.convert_tokens_to_ids(TGT_LANG)
        dtype = self.dtype or (torch.float16 if self.device in ("mps", "cuda") else torch.float32)
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

        self.tokenizer.src_lang = NLLB_CODES[language]
        batch = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=256)
        batch = batch.to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **batch,
                forced_bos_token_id=self.target_id,
                num_beams=4,
                no_repeat_ngram_size=4,  # ends the loops NLLB falls into on spoken numbers; chrF unchanged
                max_length=None,  # the model's default of 200 would otherwise conflict with max_new_tokens
                max_new_tokens=2 * batch["input_ids"].shape[1] + 16,  # English is longer than Amharic, not 2x
            )
        # Amharic has no case, so the model often starts sentences in lower case.
        sentences = (s[:1].upper() + s[1:] for s in self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return " ".join(sentences)
