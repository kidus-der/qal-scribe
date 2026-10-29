import dataclasses
import logging
import struct
import sys

import numpy as np
import pytest

from qal_scribe import decode
from qal_scribe.models import MODELS

SPEC = MODELS["amh"]
ARPA = """\\data\\
ngram 1=5
ngram 2=3

\\1-grams:
-1.0\t<unk>\t0
-99\t<s>\t-0.5
-1.0\t</s>\t0
-0.5\tab\t-0.3
-1.5\tba\t-0.3

\\2-grams:
-0.2\t<s> ab\t0
-0.3\tab </s>\t0
-0.4\t<s> ba\t0

\\end\\
"""


class FakeProcessor:
    def batch_decode(self, ids):
        return ["[AMH] ሰላም   ዓለም "]


def test_greedy_strips_the_output_tag_and_extra_spaces():
    assert decode.greedy(np.zeros((3, 4)), FakeProcessor(), SPEC) == "ሰላም ዓለም"


def test_decode_without_lm_is_greedy():
    logits = np.zeros((3, 4), dtype="float32")
    assert decode.decode(logits, FakeProcessor(), SPEC) == "ሰላም ዓለም"


def test_decode_uses_the_lm_when_given():
    class Lm:
        def decode(self, logits):
            return "from the lm"

    assert decode.decode(np.zeros((3, 4)), FakeProcessor(), SPEC, Lm()) == "from the lm"


def test_load_lm_is_none_when_the_lm_file_is_missing(tmp_path):
    assert decode.load_lm(tmp_path, SPEC) is None


def test_load_lm_is_none_when_the_model_has_no_lm(tmp_path):
    assert decode.load_lm(tmp_path, dataclasses.replace(SPEC, lm_file=None)) is None


def test_load_lm_is_none_without_the_decoder_library(tmp_path, monkeypatch):
    (tmp_path / "lm").mkdir()
    (tmp_path / SPEC.lm_file).write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "flashlight.lib.text.decoder.kenlm", None)  # makes the import fail
    assert decode.load_lm(tmp_path, SPEC) is None


def test_load_lm_is_none_for_an_unreadable_lm(tmp_path, caplog):
    pytest.importorskip("flashlight.lib.text.decoder.kenlm")
    (tmp_path / "lm").mkdir()
    (tmp_path / SPEC.lm_file).write_bytes(b"not a language model")
    with caplog.at_level(logging.WARNING):
        assert decode.load_lm(tmp_path, SPEC) is None
    assert "greedy" in caplog.text


def test_vocabulary_is_read_from_the_end_of_a_binary(tmp_path):
    lm = tmp_path / "x.bin"
    header = bytes(108) + struct.pack("<Q", 4)  # the unigram count, where a KenLM binary keeps it
    lm.write_bytes(header + b"<unk>\0ab\0ba\0x<unk>\0")  # a word may itself end in "<unk>"
    assert decode._vocabulary(lm) == ["<unk>", "ab", "ba", "x<unk>"]


def test_vocabulary_refuses_a_file_whose_word_count_does_not_match(tmp_path):
    lm = tmp_path / "x.bin"
    lm.write_bytes(bytes(108) + struct.pack("<Q", 9) + b"<unk>\0ab\0")
    with pytest.raises(ValueError):
        decode._vocabulary(lm)


@pytest.fixture
def arpa(tmp_path):
    pytest.importorskip("flashlight.lib.text.decoder.kenlm")
    path = tmp_path / "toy.arpa"
    path.write_text(ARPA, encoding="utf-8")
    return path


def test_frequent_keeps_words_the_lm_gives_more_than_the_floor(arpa, monkeypatch):
    monkeypatch.setattr(decode, "MIN_UNIGRAM", -1.0)
    assert decode._frequent(arpa, ["ab", "ba"]) == ["ab"]


# tokens: 0 " ", 1 "a", 2 "b", 3 blank, 4 a language tag (anything that is not a letter counts as blank)
CHARS = {0: " ", 1: "a", 2: "b"}


def frames(*rows):
    """Logits from per-frame probabilities over the 5 tokens above."""
    return np.log(np.array(rows, dtype="float32") + 1e-6)


BLANK = [0, 0, 0, 1, 0]
A_OR_B = [0, 0.4, 0.6, 0, 0]  # the acoustic model leans to "b" here
B_OR_A = [0, 0.6, 0.4, 0, 0]  # and to "a" here: "ba" beats "ab" acoustically, but the LM only knows "ab"


@pytest.fixture
def beam_lm(arpa):
    from flashlight.lib.text.decoder.kenlm import KenLM
    from flashlight.lib.text.dictionary import Dictionary

    words = Dictionary()
    for w in ("ab", "ba", "<unk>"):
        words.add_entry(w)
    words.set_default_index(words.get_index("<unk>"))
    return decode.BeamLM(KenLM(str(arpa), words), words, CHARS, alpha=0.0, beta=0.0)


def test_without_lm_weight_the_acoustic_model_decides(beam_lm):
    assert beam_lm.decode(frames(A_OR_B, BLANK, B_OR_A, BLANK)) == "ba"


def test_the_lm_can_overrule_the_acoustic_model(beam_lm):
    beam_lm.alpha = 1.0
    assert beam_lm.decode(frames(A_OR_B, BLANK, B_OR_A, BLANK)) == "ab"


def test_spaces_separate_words_and_stray_spaces_are_ignored(beam_lm):
    sure_a, sure_b, space = [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]
    rows = [space, BLANK, sure_a, BLANK, sure_b, BLANK, space, BLANK, space, BLANK]  # a stray leading space
    rows += [sure_b, BLANK, sure_a, space]
    assert beam_lm.decode(frames(*rows)) == "ab ba"


def test_silence_and_tags_decode_to_nothing(beam_lm):
    tag = [0, 0, 0, 0, 1]
    assert beam_lm.decode(frames(BLANK, BLANK)) == ""
    assert beam_lm.decode(frames(tag, tag, BLANK)) == ""
    assert beam_lm.decode(np.zeros((0, 5), dtype="float32")) == ""


def test_the_lm_does_not_glue_unknown_words_into_one(beam_lm):
    beam_lm.alpha, beam_lm.beta = 1.0, 0.0
    assert beam_lm._unblob(["ab", "baab"], ["ab", "ba", "ab"]) == ["ab", "ba", "ab"]  # baab is unknown
    assert beam_lm._unblob(["ab"], ["a", "b"]) == ["ab"]  # a known word stays, even if greedy split it
    assert beam_lm._unblob(["bbb"], ["ab", "ba"]) == ["bbb"]  # unknown, but no greedy words make it


def test_greedy_words_merge_repeats_and_drop_blanks(beam_lm):
    sure_a, sure_b, space = [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]
    rows = [sure_a, sure_a, BLANK, sure_a, sure_b, space, [0, 0, 0, 0, 1], sure_b]
    assert beam_lm._greedy(frames(*rows)) == ["aab", "b"]


class PauseProcessor:
    class tokenizer:  # stands in for processor.tokenizer
        word_delimiter_token = "|"

        @staticmethod
        def get_vocab():
            return {"|": 0, "a": 1, "b": 2, "[PAD]": 3, "[AMH]": 4}


def test_split_pauses_cuts_in_the_middle_of_long_runs_without_letters():
    a, gap, blank = [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], BLANK
    rows = [a, gap, a] + [blank] * 4 + [gap] * 4 + [a, a] + [blank] * 3 + [a]
    pieces = decode.split_pauses(frames(*rows), PauseProcessor(), min_frames=6)
    assert [len(p) for p in pieces] == [7, 10]  # one pause of 8 frames, cut at its middle; the 3-frame one stays
    assert np.array_equal(np.concatenate(pieces), frames(*rows))
    assert len(decode.split_pauses(frames(*[blank] * 20), PauseProcessor())) == 1  # silence alone: nothing to cut
