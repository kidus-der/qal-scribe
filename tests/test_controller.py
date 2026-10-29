import threading
import time

import numpy as np
import pytest

from qal_scribe import config, controller, models
from qal_scribe import events as ev
from qal_scribe.audio import Recorder
from qal_scribe.engine import Result

AUDIO = np.full(16000, 0.1, dtype="float32")


class FakeEngine:
    """Stands in for Engine: slow preview/transcribe on demand, records what was asked of it."""

    def __init__(self, spec=None, preview_seconds=0.0, transcribe_seconds=0.0, previews=("draft",)):
        self.device, self.lm, self.spec = "cpu", None, spec
        self.preview_seconds, self.transcribe_seconds = preview_seconds, transcribe_seconds
        self.previews = list(previews)
        self.preview_calls = 0
        self.preview_started, self.transcribing = threading.Event(), threading.Event()
        self.unloaded = False
        self.fail_load = False

    def load(self, on_progress=lambda _: None):
        on_progress("fetching")
        if self.fail_load:
            raise OSError("no network")

    def unload(self):
        self.unloaded = True

    def transcribe(self, audio):
        self.transcribing.set()
        time.sleep(self.transcribe_seconds)
        return Result(" ሰላም   ዓለም ", self.transcribe_seconds)

    def preview(self, audio):
        self.preview_calls += 1
        self.preview_started.set()
        time.sleep(self.preview_seconds)
        text = self.previews.pop(0) if len(self.previews) > 1 else self.previews[0]
        return Result(text, self.preview_seconds)


class FakeRecorder:
    def __init__(self):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
        return AUDIO

    def snapshot(self):
        return AUDIO


class FakeHotkeys:
    stopped = False

    def stop(self):
        self.stopped = True


def make(monkeypatch, tmp_path, pasted=True, **engine_kw):
    cfg = config.Config(words_file=tmp_path / "words.txt", sounds=False)
    out = []
    c = controller.Controller(cfg, out.append)
    c.engine, c.recorder = FakeEngine(models.get("amh"), **engine_kw), FakeRecorder()
    monkeypatch.setattr(controller.paste, "paste", lambda text, delay: pasted)
    monkeypatch.setattr(controller, "PREVIEW_INTERVAL", 0.01)
    return c, out


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def kinds(out):
    return [type(e).__name__ for e in out]


def test_handle_emits_transcribed(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    c.handle("dictate", AUDIO)
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


def test_ready_reports_whether_the_lm_is_active(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path)
    assert c._ready().lm is False
    c.engine.lm = object()
    ready = c._ready()
    assert ready.lm is True and ready.language == "amh" and ready.hotkeys == {"Right Option": "dictate",
                                                                              "Right Control": "translate"}


def test_toggle_stop_keeps_the_mode_the_recording_started_with(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    c.hotkeys = FakeHotkeys()
    c.toggle("translate")
    c.toggle()  # a plain `qal-scribe toggle` ends the translate recording
    assert c._clips.get_nowait()[0] == "translate"
    assert [e.mode for e in out if isinstance(e, ev.RecordingStopped)] == ["translate"]


def test_toggle_while_loading_is_refused(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="still loading"):
        c.toggle()
    assert not c.recorder.active


def test_start_checks_mic_and_single_instance_before_loading_the_model(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path)
    order = []

    def second_instance(on_toggle):
        order.append("serve")
        raise RuntimeError("qal-scribe is already running")

    monkeypatch.setattr(controller, "Recorder", lambda **kw: order.append("recorder") or FakeRecorder())
    monkeypatch.setattr(controller.ipc, "serve", second_instance)
    with pytest.raises(RuntimeError, match="already running"):
        c.start()
    assert order == ["recorder", "serve"] and c.hotkeys is None


def test_recorder_snapshot_reads_without_stopping():
    r = Recorder.__new__(Recorder)
    r._chunks = []
    assert len(r.snapshot()) == 0
    r._chunks = [np.ones(3, dtype="float32"), np.zeros(2, dtype="float32")]
    assert r.snapshot().tolist() == [1, 1, 1, 0, 0] and len(r._chunks) == 2


# --- live preview ---------------------------------------------------------------------------------------------


def test_preview_emits_only_changed_text_while_recording(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path, previews=("ሰላም", "ሰላም", "ሰላም ዓለም"))
    c._start_recording("dictate")
    wait_for(lambda: kinds(out).count("Partial") == 2)
    c._stop_recording("dictate")
    assert [e.text for e in out if isinstance(e, ev.Partial)] == ["ሰላም", "ሰላም ዓለም"]


def test_preview_off_by_config(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    c.cfg.preview = False
    c._start_recording("dictate")
    time.sleep(0.1)
    c._stop_recording("dictate")
    assert c.engine.preview_calls == 0 and "Partial" not in kinds(out)


def test_preview_skips_silence_and_stub_audio(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path)
    for quiet in (np.zeros(16000, dtype="float32"), AUDIO[:1000]):
        c.recorder.snapshot = lambda quiet=quiet: quiet
        c._start_recording("dictate")
        time.sleep(0.1)
        c._stop_recording("dictate")
    assert c.engine.preview_calls == 0


def test_final_waits_at_most_one_preview_and_the_stale_preview_is_dropped(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path, preview_seconds=0.4)
    c._start_recording("dictate")
    assert c.engine.preview_started.wait(3)  # a preview is now in flight
    c._stop_recording("dictate")
    mode, audio = c._clips.get_nowait()
    t = time.perf_counter()
    c.handle(mode, audio)
    waited = time.perf_counter() - t
    assert waited < 0.4 + 0.25
    time.sleep(0.2)  # give the in-flight preview every chance to emit late
    assert kinds(out) == ["RecordingStarted", "RecordingStopped", "Transcribed"]
    assert c.engine.preview_calls == 1  # and no new preview started once recording stopped


def test_preview_that_loses_the_race_with_stop_never_touches_the_model(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    entered, gate = threading.Event(), threading.Event()

    def slow_snapshot():
        entered.set()
        gate.wait(3)
        return AUDIO

    c.recorder.snapshot = slow_snapshot
    c._start_recording("dictate")
    assert entered.wait(3)  # the preview thread is between "recording" and "ask the model"
    c._stop_recording("dictate")
    gate.set()
    time.sleep(0.1)
    assert c.engine.preview_calls == 0 and "Partial" not in kinds(out)


def test_final_is_not_delayed_when_no_preview_is_running(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path)
    c.cfg.preview = False
    t = time.perf_counter()
    c.handle("dictate", AUDIO)
    assert time.perf_counter() - t < 0.2


def test_no_preview_starts_while_a_final_is_running(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path, transcribe_seconds=0.4)
    final = threading.Thread(target=c.handle, args=("dictate", AUDIO))
    final.start()
    assert c.engine.transcribing.wait(3)
    c._start_recording("dictate")  # next dictation begins while the previous clip is still transcribing
    time.sleep(0.2)
    assert c.engine.preview_calls == 0
    final.join()
    wait_for(lambda: c.engine.preview_calls > 0)  # previews resume once the GPU is free
    c._stop_recording("dictate")


def test_slow_previews_give_up_for_this_recording(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path, previews=("a", "b", "c"))
    monkeypatch.setattr(controller, "PREVIEW_MAX_SECONDS", -1)  # every preview counts as too slow
    c._start_recording("dictate")
    wait_for(lambda: c.engine.preview_calls == 1)
    time.sleep(0.15)
    c._stop_recording("dictate")
    assert c.engine.preview_calls == 1 and kinds(out).count("Partial") == 1


def test_preview_error_becomes_one_warning_and_recording_goes_on(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)

    def boom(audio):
        c.engine.preview_calls += 1
        raise RuntimeError("mps out of memory")

    c.engine.preview = boom
    c._start_recording("dictate")
    wait_for(lambda: "Notice" in kinds(out))
    time.sleep(0.1)
    c._stop_recording("dictate")
    (notice,) = [e for e in out if isinstance(e, ev.Notice)]
    assert notice.level == "warning" and "mps out of memory" in notice.message and c.engine.preview_calls == 1


# --- language switch ------------------------------------------------------------------------------------------


@pytest.fixture
def two_languages(monkeypatch):
    other = models.ModelSpec("xyz", "Xyzish", "x/y", "rev", "MIT", "https://example.org")
    monkeypatch.setitem(models.MODELS, "xyz", other)
    return other


def test_set_language_loads_in_background_then_swaps_and_frees_the_old_model(monkeypatch, tmp_path, two_languages):
    c, out = make(monkeypatch, tmp_path)
    old, new = c.engine, FakeEngine(two_languages)
    new.lm = object()
    monkeypatch.setattr(controller, "Engine", lambda spec, device, use_lm: new)
    c.set_language("xyz")
    wait_for(lambda: "Ready" in kinds(out))
    assert kinds(out)[0] == "Loading" and kinds(out)[-1] == "Ready"
    ready = out[-1]
    assert ready.language == "xyz" and ready.model == "x/y" and ready.lm is True
    assert c.engine is new and old.unloaded and not new.unloaded
    assert c.spec is two_languages and c.cfg.language == "xyz"


def test_old_model_serves_until_the_new_one_is_ready(monkeypatch, tmp_path, two_languages):
    c, out = make(monkeypatch, tmp_path)
    old, new, release = c.engine, FakeEngine(two_languages), threading.Event()
    new.load = lambda on_progress=None: release.wait(3)
    monkeypatch.setattr(controller, "Engine", lambda spec, device, use_lm: new)
    c.set_language("xyz")
    wait_for(lambda: "Loading" in kinds(out))
    c.handle("dictate", AUDIO)  # dictation keeps working while loading
    assert c.engine is old and not old.unloaded
    release.set()
    wait_for(lambda: "Ready" in kinds(out))
    assert c.engine is new


def test_failed_switch_keeps_the_old_model(monkeypatch, tmp_path, two_languages):
    c, out = make(monkeypatch, tmp_path)
    old, new = c.engine, FakeEngine(two_languages)
    new.fail_load = True
    monkeypatch.setattr(controller, "Engine", lambda spec, device, use_lm: new)
    c.set_language("xyz")
    wait_for(lambda: "Notice" in kinds(out))
    assert out[-1].level == "error" and "Xyzish" in out[-1].message
    assert c.engine is old and not old.unloaded and c.cfg.language == "amh"


def test_switch_to_the_current_language_just_reports_ready(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    c.set_language("amh")
    wait_for(lambda: out)
    assert kinds(out) == ["Ready"] and not c.engine.unloaded


def test_unknown_language_is_rejected_immediately(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no model for language"):
        c.set_language("nope")
    assert not out


def test_second_switch_while_one_is_running_is_refused(monkeypatch, tmp_path, two_languages):
    c, out = make(monkeypatch, tmp_path)
    new, release = FakeEngine(two_languages), threading.Event()
    new.load = lambda on_progress=None: release.wait(3)
    monkeypatch.setattr(controller, "Engine", lambda spec, device, use_lm: new)
    c.set_language("xyz")
    wait_for(lambda: "Loading" in kinds(out))
    c.set_language("xyz")
    wait_for(lambda: "Notice" in kinds(out))
    release.set()
    wait_for(lambda: "Ready" in kinds(out))
    assert sum(isinstance(e, ev.Notice) for e in out) == 1


# --- shutdown -------------------------------------------------------------------------------------------------


def start_worker(c):
    worker = threading.Thread(target=c._worker, daemon=True)
    worker.start()
    return worker


def test_stop_releases_everything(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path)
    c.hotkeys = FakeHotkeys()
    ipc_stopped = []
    monkeypatch.setattr(controller.ipc, "stop", lambda: ipc_stopped.append(True), raising=False)
    worker = start_worker(c)
    c._start_recording("dictate")  # still recording when asked to quit
    c.translator = object()
    c.stop()
    worker.join(3)
    assert c.hotkeys.stopped and ipc_stopped and not c.recorder.active and c._preview_stop.is_set()
    assert c.engine.unloaded and c.translator is None and not worker.is_alive()


def test_stop_waits_for_the_model_call_in_flight_and_drops_queued_clips(monkeypatch, tmp_path):
    c, out = make(monkeypatch, tmp_path, transcribe_seconds=0.2)
    worker = start_worker(c)
    c._clips.put(("dictate", AUDIO))
    c._clips.put(("dictate", AUDIO))
    assert c.engine.transcribing.wait(3)
    c.stop()  # returns only after the running transcription released the model
    worker.join(3)
    assert c.engine.unloaded and kinds(out) == ["Transcribed"]


def test_stop_does_not_hang_on_a_stuck_model_call(monkeypatch, tmp_path):
    c, _ = make(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "STOP_WAIT", 0.05)
    with c._gpu:  # a model call that never ends
        c.stop()
    assert not c.engine.unloaded


def test_stop_before_start_is_safe(monkeypatch, tmp_path):
    cfg = config.Config(words_file=tmp_path / "words.txt")
    c = controller.Controller(cfg, lambda e: None)
    c.engine = FakeEngine()
    c.stop()
    assert c.engine.unloaded
