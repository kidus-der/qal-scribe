"""Translator behaviour with the model mocked: no weights are loaded."""
import huggingface_hub
import pytest
import torch
import transformers

from qal_scribe import translate
from qal_scribe.translate import MAX_WORDS, Translator, split_sentences


def test_split_on_ethiopic_and_latin_marks():
    assert split_sentences("ሰላም ነው።ደህና ነህ፧ አዎ. እሺ! መልካም") == ["ሰላም ነው።", "ደህና ነህ፧", "አዎ.", "እሺ!", "መልካም"]


def test_split_chunks_text_without_punctuation():
    words = ["ቃል"] * (MAX_WORDS * 2 + 5)
    chunks = split_sentences(" ".join(words))
    assert [len(c.split()) for c in chunks] == [MAX_WORDS, MAX_WORDS, 5]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_split_ignores_blank(text):
    assert split_sentences(text) == []


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __call__(self, sentences, **kw):
        self.sentences = sentences
        return FakeBatch(input_ids=torch.zeros((len(sentences), 7), dtype=torch.long))

    def batch_decode(self, out, skip_special_tokens):
        return [f"line {i} of {len(out)}" if i else "first lower" for i in range(len(out))]


class FakeModel:
    dtype = torch.float32

    def generate(self, **kw):
        self.kw = kw
        return torch.zeros((kw["input_ids"].shape[0], 3), dtype=torch.long)


def loaded_translator() -> Translator:
    t = Translator("cpu")
    t.model, t.tokenizer, t.target_id = FakeModel(), FakeTokenizer(), 42
    return t


def test_blank_input_does_not_load():
    t = Translator("cpu")
    assert t.translate("  \n") == ""
    assert t.model is None


def test_translate_batches_sentences_and_capitalises():
    t = loaded_translator()
    out = t.translate("ሰላም ነው። ደህና ነህ፧")
    assert t.tokenizer.sentences == ["ሰላም ነው።", "ደህና ነህ፧"]
    assert out == "First lower Line 1 of 2"
    kw = t.model.kw
    assert kw["num_beams"] == 4 and kw["forced_bos_token_id"] == 42 and kw["no_repeat_ngram_size"] == 4
    assert kw["max_new_tokens"] == 2 * 7 + 16 and kw["max_length"] is None


@pytest.fixture
def fake_hub(monkeypatch):
    """Stub the download and the transformers loaders; records the dtype the model was requested in."""
    calls = {}

    class Model:
        def __init__(self, fail_on=None):
            self.fail_on = fail_on

        def eval(self):
            return self

        def to(self, device):
            if device == self.fail_on:
                raise RuntimeError("no such device")
            calls["device"] = device
            return self

        def float(self):
            calls["float"] = True
            return self

    def from_pretrained(path, dtype):
        calls["dtype"] = dtype
        return Model(fail_on="mps")

    class Tokenizer:
        @staticmethod
        def from_pretrained(path, src_lang):
            calls["src_lang"] = src_lang
            tok = FakeTokenizer()
            tok.convert_tokens_to_ids = lambda t: calls.setdefault("target", t) and 7
            return tok

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: calls.update(repo=repo, **kw) or "/x")
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(transformers, "AutoTokenizer", Tokenizer)
    return calls


def test_load_pins_revision_and_languages(fake_hub):
    t = Translator("cpu")
    t.load()
    assert fake_hub["repo"] == Translator.MODEL and fake_hub["revision"] == Translator.REVISION
    assert fake_hub["src_lang"] == translate.SRC_LANG and fake_hub["target"] == translate.TGT_LANG
    assert fake_hub["dtype"] == torch.float32 and fake_hub["device"] == "cpu"


def test_load_uses_float16_on_gpu_and_falls_back_to_cpu(fake_hub):
    t = Translator("mps")
    t.load()
    assert fake_hub["dtype"] == torch.float16
    assert t.device == "cpu" and fake_hub["device"] == "cpu" and fake_hub["float"]


def test_load_is_idempotent(fake_hub):
    t = Translator("cpu")
    t.load()
    model = t.model
    fake_hub.clear()
    t.load()
    assert t.model is model and not fake_hub


def test_a_list_is_taken_as_its_sentences():
    t = loaded_translator()
    t.translate(["ሰላም ነው", "ደህና ነህ፧ አዎ.", "  "])
    assert t.tokenizer.sentences == ["ሰላም ነው", "ደህና ነህ፧", "አዎ."]


def test_the_source_language_follows_the_dictation_language():
    t = loaded_translator()
    t.translate("ሰላም", language="tir")
    assert t.tokenizer.src_lang == "tir_Ethi"
    t.translate("nagaa", language="orm")
    assert t.tokenizer.src_lang == "gaz_Latn"
    with pytest.raises(ValueError, match="no translation model for language 'sid'"):
        t.translate("ሰላም", language="sid")
