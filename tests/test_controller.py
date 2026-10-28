import numpy as np

from qal_scribe import config, controller
from qal_scribe import events as ev
from qal_scribe.engine import Result


class FakeEngine:
    device, lm = "cpu", None

    def transcribe(self, audio):
        return Result(" ሰላም   ዓለም ", 0.01)


def make(monkeypatch, tmp_path, pasted=True):
    cfg = config.Config(words_file=tmp_path / "words.txt")
    out = []
    c = controller.Controller(cfg, out.append)
    c.engine = FakeEngine()
    monkeypatch.setattr(controller.paste, "paste", lambda text, delay: pasted)
    return c, out


def test_handle_emits_transcribed(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    c.handle("dictate", np.full(16000, 0.1, dtype="float32"))
    (t,) = out
    assert isinstance(t, ev.Transcribed) and t.text == "ሰላም ዓለም" and t.pasted and t.audio_seconds == 1.0


def test_silence_is_skipped(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    c.handle("dictate", np.zeros(16000, dtype="float32"))
    assert isinstance(out[0], ev.Notice)


def test_words_file_parsing(tmp_path):
    p = tmp_path / "words.txt"
    p.write_text("# comment\nከርሰር => Cursor\n\nኤፒአይ=>API # inline\n", encoding="utf-8")
    assert controller.load_words(config.Config(words_file=p)) == {"ከርሰር": "Cursor", "ኤፒአይ": "API"}
