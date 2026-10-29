"""`qal-scribe --plain` with a stand-in Controller: no model, no audio, no hotkeys."""
import pytest

from qal_scribe import events as ev
from qal_scribe import plain
from qal_scribe.config import Config


def test_ready_without_hotkeys_says_toggle():
    line = plain.format_event(ev.Ready("cpu", "m", "amh", False, {}))
    assert "hold" not in line and "run `qal-scribe toggle` (bound to a key)" in line


@pytest.mark.parametrize("stopped_cleanly", [True, False])
def test_ctrl_c_exits_hard_only_when_a_model_call_is_still_running(monkeypatch, stopped_cleanly):
    class Controller:
        def __init__(self, cfg, emit):
            pass

        def start(self):
            raise KeyboardInterrupt

        def stop(self):
            return stopped_cleanly

    exits = []
    monkeypatch.setattr(plain, "Controller", Controller)
    monkeypatch.setattr(plain.os, "_exit", exits.append)
    plain.run(Config())
    assert exits == ([] if stopped_cleanly else [0])


def test_a_controller_that_cannot_be_built_is_one_line(monkeypatch):
    def unreadable(cfg, emit):
        raise ValueError("words.txt: not UTF-8 text")

    monkeypatch.setattr(plain, "Controller", unreadable)
    with pytest.raises(SystemExit, match="qal-scribe: words.txt: not UTF-8 text"):
        plain.run(Config())
