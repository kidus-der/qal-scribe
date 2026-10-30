import subprocess
import sys

import pytest

from qal_scribe import events as ev
from qal_scribe import indicator


class FakeProc:
    def __init__(self, broken=False):
        self.lines, self.broken, self.closed = [], broken, False
        self.stdin = self

    def write(self, text):
        if self.broken:
            raise BrokenPipeError
        self.lines += text.split()

    def flush(self):
        pass

    def close(self):
        self.closed = True


def make(proc=None):
    proc, errors = proc or FakeProc(), []
    return indicator.Indicator(errors.append, popen=lambda *a, **kw: proc), proc, errors


def test_states_follow_a_recording():
    ind, proc, errors = make()
    for e in (ev.RecordingStarted("dictate"), ev.Level(0.5), ev.RecordingStopped("dictate", 2.0),
              ev.Partial("x"), ev.Transcribed("dictate", "t", "r", 2.0, 0.5, True)):
        ind(e)
    assert proc.lines == ["listening", "transcribing", "hidden"] and not errors


def test_a_new_recording_wins_over_the_previous_clip_finishing():
    ind, proc, _ = make()
    for e in (ev.RecordingStarted("dictate"), ev.RecordingStopped("dictate", 1.0), ev.RecordingStarted("dictate"),
              ev.Transcribed("dictate", "t", "r", 1.0, 0.5, True)):
        ind(e)
    assert proc.lines == ["listening", "transcribing", "listening"]  # still listening after the first clip ends
    ind(ev.RecordingStopped("dictate", 1.0))
    ind(ev.Notice("too short", "warning", clip=True))
    assert proc.lines[-2:] == ["transcribing", "hidden"]


def test_a_clip_that_fails_hides_the_pill_but_other_notices_do_not():
    ind, proc, _ = make()
    ind(ev.RecordingStarted("translate"))
    ind(ev.RecordingStopped("translate", 1.0))
    ind(ev.Notice("loading", "info"))
    assert proc.lines == ["listening", "transcribing"]
    ind(ev.Notice("boom", "error", clip=True))
    assert proc.lines[-1] == "hidden"


def test_a_helper_that_dies_gives_one_notice_and_goes_quiet():
    ind, _, errors = make(FakeProc(broken=True))
    ind(ev.RecordingStarted("dictate"))
    ind(ev.RecordingStopped("dictate", 1.0))
    assert len(errors) == 1 and "dictation works without it" in errors[0]


def test_a_helper_that_cannot_start_gives_one_notice():
    errors = []

    def no_python(*a, **kw):
        raise FileNotFoundError("python")

    ind = indicator.Indicator(errors.append, popen=no_python)
    ind(ev.RecordingStarted("dictate"))
    assert len(errors) == 1 and "FileNotFoundError" in errors[0]


def test_close_closes_the_helpers_stdin_and_later_events_are_ignored():
    ind, proc, _ = make()
    ind.close()
    ind(ev.RecordingStarted("dictate"))
    assert proc.closed and proc.lines == []


@pytest.mark.skipif(sys.platform != "darwin", reason="AppKit helper")
def test_the_helper_exits_when_stdin_closes():
    p = subprocess.Popen([sys.executable, "-m", "qal_scribe.indicator"], stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    p.stdin.write("bogus\n")  # unknown lines are ignored, and nothing is shown without a state
    p.stdin.close()
    assert p.wait(10) == 0, p.stderr.read()
