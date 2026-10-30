"""TUI behaviour, driven by a FakeController that emits scripted events: no model, no audio, no hotkeys."""
import asyncio
import dataclasses
import io
import os
import re
import subprocess
import sys
import threading
import types
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console
from textual.worker import WorkerFailed

from qal_scribe import app as app_mod
from qal_scribe import events as ev
from qal_scribe import models
from qal_scribe.app import QalApp
from qal_scribe.config import Config
from qal_scribe.hotkey import WAYLAND_REASON
from qal_scribe.tui.widgets import HistoryRow, Meter

KEYS = {"Right Option": "dictate", "Right Control": "translate"}
READY = ev.Ready("mps", "snapwre/hohe-asr-amharic", "amh", True, KEYS)
SAMPLE = "ሰላም ዓለም፣ እንዴት ነህ? ዛሬ ጥሩ ቀን ነው።"


class FakeController:
    """Same surface as Controller (plus the optional stop/set_language), nothing real behind it."""

    def __init__(self, cfg, emit, hold=False):
        self.cfg, self.emit = cfg, emit
        self.hold = threading.Event() if hold else None  # start() blocks until released: the loading state
        self.started = threading.Event()
        self.stopped = False
        self.languages: list[str] = []

    def start(self):
        self.started.set()
        if self.hold:
            self.hold.wait(10)
        self.emit(READY)

    def stop(self):
        self.stopped = True

    def set_language(self, code):
        self.languages.append(code)


COPIED: list[str] = []  # what the copy key put on the (fake) system clipboard


def drive(scenario, *, size=(120, 36), cfg=None, hold=False, factory=None, clipboard=True, notifications=False,
          setup=False):
    """Run `scenario(app, fake, pilot)` against the app (fake = the FakeController it started).
    clipboard=False: the system clipboard is unreachable. notifications=True draws toasts (screenshots).
    setup=True: the setup screen opens at start; False: it never does, whatever this machine has cached;
    None: only when something is missing, as in real use."""
    made: list[FakeController] = []

    def make(c, emit):
        made.append(FakeController(c, emit, hold=hold))
        return made[0]

    async def go():
        app = QalApp(cfg or Config(), controller_factory=factory or make, setup=setup,
                     config_path=Path(os.environ["XDG_CONFIG_HOME"]) / "qal-scribe" / "config.toml")
        async with app.run_test(size=size, notifications=notifications) as pilot:
            if not hold and setup is False:
                await settle(pilot)
            await scenario(app, made[0] if made else None, pilot)

    COPIED.clear()
    with (mock.patch.object(app_mod, "mic_name", lambda device=None: "Test Microphone"),  # never open the audio stack
          mock.patch("qal_scribe.paste.copy", lambda text: COPIED.append(text) or clipboard)):  # nor the clipboard
        asyncio.run(go())


async def settle(pilot):
    """Wait for the controller start and every worker, then for the messages they posted."""
    await asyncio.to_thread(pilot.app.boot_thread.join, 5)
    try:
        await pilot.app.workers.wait_for_complete()
    except WorkerFailed:
        pass
    await pilot.pause()


async def send(pilot, *events):
    app = pilot.app
    for e in events:
        app.post_message(app_mod.ControllerEvent(e))
    await pilot.pause()


def screen_text(app) -> str:
    console = Console(width=app.size.width, height=app.size.height, file=io.StringIO(), force_terminal=True,
                      color_system="truecolor", record=True)
    console.print(app.screen._compositor.render_update(full=True))
    return console.export_text()


def shown(app, selector: str) -> str:
    return str(app.query_one(selector).render())


def pill(app) -> str:
    return shown(app, "#pill")


def toasts(app) -> list[tuple[str, str]]:
    return [(n.message, n.severity) for n in app._notifications]


def done(text=SAMPLE, mode="dictate", pasted=True, audio=3.2, compute=0.13):
    return ev.Transcribed(mode, text, text, audio, compute, pasted)


def rows(app) -> list[str]:
    return [r.entry.text for r in app.query(HistoryRow)]


def two_languages(monkeypatch):
    spec = models.MODELS["amh"]
    swahili = dataclasses.replace(spec, language="swh", language_name="Swahili")
    monkeypatch.setattr(models, "MODELS", {"amh": spec, "swh": swahili})
    monkeypatch.setattr(app_mod, "installed_languages", lambda current: list(models.MODELS))


def test_installed_languages_skips_models_that_are_not_downloaded(monkeypatch):
    monkeypatch.setattr("qal_scribe.doctor.is_cached", lambda repo, *_: repo == models.MODELS["tir"].repo)
    assert app_mod.installed_languages("amh") == ["amh", "tir"]
    assert app_mod.installed_languages("som") == ["tir", "som"]


def test_importing_the_app_does_not_load_torch():
    code = "import sys, qal_scribe.app; sys.exit(any(m in sys.modules for m in ('torch', 'transformers')))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# -- loading and failure ----------------------------------------------------------------------------------


def test_paints_while_the_model_loads():
    async def scenario(app, fake, pilot):
        assert fake.started.wait(5)
        await send(pilot, ev.Loading("downloading model 40%"))
        assert "LOADING" in pill(app)
        text = screen_text(app)
        assert "downloading model 40%" in text and "━" in text
        assert app.screen.has_class("-loading")
        await pilot.press("m")  # the UI thread is free while start() blocks
        assert app.screen.has_class("-compact")
        fake.hold.set()
        await settle(pilot)
        assert pill(app) == "READY"

    drive(scenario, hold=True)


def test_ready_shows_model_device_mic_and_the_empty_states():
    async def scenario(app, fake, pilot):
        text = screen_text(app)
        assert "ቃል" in text and "hohe-asr-amharic" in text and "mps" in text and "LM on" in text
        assert "mic Test Microphone" in text and "Amharic" in text
        assert "Hold Right Option to dictate Amharic" in text
        assert "Hold Right Control to translate to English" in text
        assert "Your dictations will collect here" in text
        assert pill(app) == "READY" and app.screen.has_class("-ready")

    drive(scenario)


def test_model_progress_bars_in_the_boot_thread_do_not_break_the_start(tmp_path):
    """Textual's stderr has no file descriptor; the first tqdm bar starts multiprocessing's resource tracker,
    which needs one. A fresh interpreter, so neither the tracker nor tqdm's lock exist yet (as in real use)."""
    script = tmp_path / "boot.py"
    script.write_text(
        "import asyncio\n"
        "from qal_scribe import app as app_mod, events as ev\n"
        "from qal_scribe.config import Config\n"
        "app_mod.mic_name = lambda device=None: None\n"
        "class Fake:\n"
        "    def __init__(self, cfg, emit): self.emit = emit\n"
        "    def start(self):\n"
        "        from tqdm import tqdm\n"
        "        for _ in tqdm(range(3)): pass\n"
        "        self.emit(ev.Ready('cpu', 'm', 'amh', False, {}))\n"
        "    def stop(self): pass\n"
        "async def go():\n"
        "    app = app_mod.QalApp(Config(), controller_factory=Fake)\n"
        "    async with app.run_test() as pilot:\n"
        "        await asyncio.to_thread(app.boot_thread.join, 20)\n"
        "        await pilot.pause()\n"
        "        print(repr(app.failure), app.info is not None)\n"
        "asyncio.run(go())\n"
    )
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, check=False)
    assert done.stdout.strip() == "'' True", done.stdout + done.stderr


def test_a_failed_start_shows_the_error_instead_of_loading_forever():
    def broken(cfg, emit):
        raise OSError("no network")

    async def scenario(app, fake, pilot):
        await settle(pilot)
        assert pill(app) == "ERROR" and app.screen.has_class("-failed")
        assert "OSError: no network" in screen_text(app)
        assert toasts(app) == []  # the stage already says it, with the way out
        assert "run `qal-scribe doctor`" in screen_text(app)

    drive(scenario, factory=broken)


def test_a_long_start_error_fits_with_the_way_out():
    reason = ("OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them "
              "in the cached files. Check your internet connection or see how to run the library in offline mode.")

    def broken(cfg, emit):
        raise OSError(reason.removeprefix("OSError: "))

    async def scenario(app, fake, pilot):
        await settle(pilot)
        assert app.query_one("#stage").border_title == "Error"
        assert "offline mode" in screen_text(app) and "s Settings · d Setup · q Quit" in screen_text(app)

    drive(scenario, factory=broken, size=(80, 24))

    async def compact(app, fake, pilot):
        await settle(pilot)
        assert "Could not start: OSError: We couldn't" in screen_text(app)

    drive(compact, factory=broken, size=(80, 4), cfg=Config(compact=True))


def test_before_ready_the_header_names_the_language():
    async def scenario(app, fake, pilot):
        assert fake.started.wait(5)
        await pilot.pause()
        assert shown(app, "#status") == "Amharic" and app.query_one("#stage").border_title == "Loading Amharic"
        fake.hold.set()
        await settle(pilot)

    drive(scenario, hold=True)


def test_status_drops_the_mic_rather_than_cutting_it_off():
    async def scenario(app, fake, pilot):
        await pilot.pause()
        assert "mic" not in screen_text(app).splitlines()[0] and "LM on" in shown(app, "#status")
        await pilot.resize_terminal(120, 36)
        await pilot.pause()
        assert "LM on · →EN nllb-600m · mic Test Microphone" in screen_text(app).splitlines()[0]

    drive(scenario, size=(80, 20))


# -- recording, transcribing, translating ----------------------------------------------------------------


def test_recording_shows_the_timer_meter_and_live_partial():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"))
        assert pill(app).startswith("● REC 0:0") and app.screen.has_class("-recording")
        assert "listening…" in screen_text(app) and "release Right Option to finish" in screen_text(app)
        await send(pilot, ev.Level(0.5), ev.Level(0.2), ev.Partial("ሰላም"))
        assert list(app.query_one(Meter).levels)[-2:] == [0.5**0.6, 0.2**0.6]
        assert "ሰላም" in screen_text(app) and app.query_one("#stage").has_class("-live")
        app.rec_started -= 65
        app.tick()
        assert pill(app).startswith("● REC 1:0")

    drive(scenario)


@pytest.mark.parametrize(("size", "compact"), [((80, 24), False), ((80, 5), True)])
def test_a_long_live_draft_keeps_its_newest_words_in_view(size, compact):
    words = [f"ቃል{i}" for i in range(120)]

    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.Partial(" ".join(words)))
        await pilot.pause()
        app.refresh_view()  # the 0.1 s tick does this while recording
        await pilot.pause()
        text = screen_text(app)
        assert words[-1] in text and words[0] not in text

    drive(scenario, size=size, cfg=Config(compact=compact))


def test_levels_outside_a_recording_are_ignored():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.Level(0.9))
        assert max(app.query_one(Meter).levels) == 0.0

    drive(scenario)


def test_meter_draws_absolute_levels():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), *[ev.Level(1.0)] * 3, ev.Level(0.0))
        lines = [line for line in screen_text(app).splitlines() if "█" in line or "▁" in line]
        assert len(lines) == 3  # the meter is three rows tall
        assert lines[-1].rstrip(" │").endswith("███▁")

    drive(scenario)


def test_stop_shows_transcribing_then_the_result():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.RecordingStopped("dictate", 3.2))
        assert "TRANSCRIBING" in pill(app) and app.screen.has_class("-transcribing")
        assert "Transcribing…" in screen_text(app) and "Hold Right Option" not in screen_text(app)
        assert max(app.query_one(Meter).levels) == 0.0
        await send(pilot, done())
        assert pill(app) == "READY"
        text = screen_text(app)
        assert SAMPLE in text and "3.2s audio · 0.13s compute · pasted" in text
        assert rows(app) == [SAMPLE]

    drive(scenario)


def test_translate_mode_has_its_own_pill_title_and_border():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("translate"))
        assert "REC → EN" in pill(app) and "release Right Control to finish" in screen_text(app)
        assert app.query_one("#stage").border_title == "Translate → English"
        await send(pilot, ev.RecordingStopped("translate", 2.0))
        assert "TRANSLATING" in pill(app) and app.screen.has_class("-translating")
        assert "Translating…" in screen_text(app)
        await send(pilot, done("hello world", mode="translate"))
        assert pill(app) == "READY" and "→ EN" in screen_text(app)

    drive(scenario)


def test_a_clip_after_a_result_shows_transcribing_not_the_old_text():
    async def scenario(app, fake, pilot):
        def stage() -> str:
            return shown(app, "#hero") if app.query_one("#hero").display else shown(app, "#transcript")

        await send(pilot, done("አንድ"), ev.RecordingStarted("translate"), ev.RecordingStopped("translate", 2.0))
        assert "Translating…" in stage() and "አንድ" not in stage()
        await send(pilot, done("", pasted=False), ev.RecordingStarted("dictate"), ev.RecordingStopped("dictate", 1))
        assert "Transcribing…" in stage()

    drive(scenario)


def test_translation_model_loading_is_a_note_not_a_return_to_loading():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStopped("translate", 2.0), ev.Loading("loading translation model"))
        assert "TRANSLATING" in pill(app) and "loading translation model" in screen_text(app)
        await send(pilot, done("hello", mode="translate"))
        assert pill(app) == "READY" and "loading translation model" not in screen_text(app)

    drive(scenario)


def test_a_language_switch_reload_shows_loading_again():
    async def scenario(app, fake, pilot):
        await send(pilot, done("አንድ"), ev.Loading("loading Swahili"))
        assert "LOADING" in pill(app) and "loading Swahili" in screen_text(app)
        await send(pilot, READY)
        assert pill(app) == "READY" and "አንድ" in screen_text(app)

    drive(scenario)


def test_a_new_recording_wins_over_a_clip_still_being_transcribed():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.RecordingStopped("dictate", 1.0))
        await send(pilot, ev.RecordingStarted("dictate"))
        assert pill(app).startswith("● REC")
        await send(pilot, done())
        assert pill(app).startswith("● REC")
        await send(pilot, ev.RecordingStopped("dictate", 1.0), done("ሁለት"))
        assert pill(app) == "READY"

    drive(scenario)


def test_a_notice_about_something_else_does_not_end_the_clip():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.RecordingStopped("dictate", 3.0))
        await send(pilot, ev.Notice("already switching language", "warning"))
        assert "TRANSCRIBING" in pill(app)
        await send(pilot, done())
        assert pill(app) == "READY"

    drive(scenario)


def test_a_skipped_clip_does_not_leave_the_pill_on_transcribing():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.RecordingStopped("dictate", 0.1))
        await send(pilot, ev.Notice("too short or silent", "warning", clip=True))
        assert pill(app) == "READY"

    drive(scenario)


def test_nothing_recognised_and_not_pasted_are_marked():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStopped("dictate", 1.0), done("", pasted=False))
        text = screen_text(app)
        assert "(nothing recognised)" in text and "nothing to paste" in text
        assert [m.render() for m in app.query(".paste")] == ["·"]  # muted, not the warning-coloured ✗
        assert app.query_one("#stage").has_class("-empty")
        assert app.check_action("copy", ()) is None  # the footer dims it; the key does nothing

    drive(scenario)


def test_compact_marks_a_result_that_was_not_pasted():
    async def scenario(app, fake, pilot):
        await send(pilot, done("ሰላም", pasted=False))
        assert "ሰላም  ✗ not pasted" in screen_text(app)
        await send(pilot, done("ዓለም"))
        assert "not pasted" not in screen_text(app)

    drive(scenario, size=(80, 5), cfg=Config(compact=True))


def test_transcripts_with_markup_characters_render_literally():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStopped("dictate", 1.0), done("[bold]ሰላም[/] [red"))
        assert "[bold]ሰላም[/] [red" in screen_text(app)

    drive(scenario)


# -- notices become toasts ---------------------------------------------------------------------------------


class SetupWarnings(FakeController):
    """A start that cannot hear the hotkeys: Wayland (no keys at all) and a paste problem."""

    def start(self):
        self.started.set()
        self.emit(ev.Notice(WAYLAND_REASON, "warning"))
        self.emit(ev.Notice("text insertion needs wtype", "warning"))
        self.emit(dataclasses.replace(READY, hotkeys={}))


def test_without_global_hotkeys_the_screen_says_toggle_and_keeps_the_warnings():
    async def scenario(app, fake, pilot):
        text = screen_text(app)
        assert "Right Option" not in text and "Hold" not in text and toasts(app) == []
        assert "Run qal-scribe toggle (bound to a key)" in text and "qal-scribe toggle --translate" in text
        assert "Wayland does not allow" not in text  # the hero already says to bind `qal-scribe toggle`
        assert "⚠ text insertion needs wtype" in text
        await send(pilot, ev.RecordingStarted("dictate"))
        assert "run `qal-scribe toggle` again to finish" in screen_text(app)
        assert "Right Option" not in screen_text(app)

    for size in ((120, 36), (80, 24)):
        drive(scenario, size=size, factory=lambda c, e: SetupWarnings(c, e))


def test_compact_shows_notices_in_the_stage_instead_of_toasts(monkeypatch):
    async def scenario(app, fake, pilot):
        message = "could not start recording: PortAudioError: Error starting stream; `qal-scribe doctor` checks it"
        await send(pilot, ev.Notice(message, "error"))
        assert toasts(app) == [] and "could not start recording" in screen_text(app) and "READY" in pill(app)
        app.flash = (app.flash[0], 0)  # its time is up
        app.refresh_view()
        await pilot.pause()
        assert "could not start recording" not in screen_text(app)

    drive(scenario, size=(80, 5), cfg=Config(compact=True))


def test_notices_become_toasts_with_matching_severity():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.Notice("fyi", "info"), ev.Notice("careful", "warning"))
        await send(pilot, ev.Notice("broke", "error"))
        assert toasts(app) == [("fyi", "information"), ("careful", "warning"), ("broke", "error")]

    drive(scenario)


# -- history ------------------------------------------------------------------------------------------------


def test_history_is_newest_first_with_time_mode_stats_and_paste_flag():
    async def scenario(app, fake, pilot):
        await send(pilot, done("አንድ", audio=1.0, compute=0.1))
        await send(pilot, done("ሁለት", mode="translate", pasted=False, audio=2.5, compute=0.2))
        assert rows(app) == ["ሁለት", "አንድ"]
        text = screen_text(app)
        assert re.search(r"\d\d:\d\d:\d\d", text) and "→ EN" in text and "dictate" in text
        assert "2.5s · 0.20s" in text and "1.0s · 0.10s" in text and "✓" in text and "✗" in text
        assert "audio · compute" in text

    drive(scenario)


def test_a_long_entry_takes_at_most_two_history_rows():
    async def scenario(app, fake, pilot):
        await send(pilot, done(" ".join(["ረጅም"] * 200)), done("አጭር"))
        assert [r.query_one(".text").size.height for r in app.query(HistoryRow)] == [1, 2]
        text = screen_text(app)
        assert "አጭር" in text and "0.13s" not in text.split("History")[1]  # 80 columns: the text gets the stats' room
        assert "✓ pasted" in text and "audio · compute" not in text

    drive(scenario, size=(80, 24))


def test_history_keeps_only_the_most_recent_entries(monkeypatch):
    monkeypatch.setattr(app_mod, "HISTORY_LIMIT", 3)

    async def scenario(app, fake, pilot):
        await send(pilot, *[done(f"t{i}") for i in range(5)])
        assert rows(app) == ["t4", "t3", "t2"]

    drive(scenario)


# -- key bindings -------------------------------------------------------------------------------------------


def test_c_copies_the_last_transcript_to_the_system_clipboard():
    async def scenario(app, fake, pilot):
        await send(pilot, done("አንድ"), done("ሁለት"))
        await pilot.press("c")
        assert COPIED == ["ሁለት"] and app.clipboard == "ሁለት" and ("copied", "information") in toasts(app)

    drive(scenario)


def test_c_without_a_clipboard_says_so_instead_of_copied():
    async def scenario(app, fake, pilot):
        await send(pilot, done("አንድ"))
        await pilot.press("c")
        assert app.clipboard == "አንድ"  # still sent to the terminal (OSC 52)
        [(message, severity)] = toasts(app)
        assert "no clipboard tool" in message and severity == "warning"

    drive(scenario, clipboard=False)


def test_enter_copies_the_selected_history_row():
    async def scenario(app, fake, pilot):
        await send(pilot, done("አንድ"), done("ሁለት"))
        await pilot.press("down", "down", "enter")
        assert app.clipboard == "አንድ"

    drive(scenario)


def test_enter_copies_the_highlighted_row_after_new_entries_arrive(monkeypatch):
    monkeypatch.setattr(app_mod, "HISTORY_LIMIT", 4)

    async def scenario(app, fake, pilot):
        await send(pilot, done("one"), done("two"), done("three"))
        await pilot.press("down")  # highlights "three", the newest
        await send(pilot, done("four"))
        history = app.query_one("#history")
        assert history.highlighted_child.entry.text == "three"
        assert [r.entry.text for r in app.query(HistoryRow) if r.has_class("-highlight")] == ["three"]
        await pilot.press("enter")
        assert app.clipboard == "three"
        await pilot.press("down", "down")  # "one", the oldest, which the next entry pushes out
        await send(pilot, done("five"))
        assert rows(app) == ["five", "four", "three", "two"] and history.highlighted_child.entry.text == "two"

    drive(scenario)


def test_l_with_one_model_says_so_and_does_not_call_the_controller(monkeypatch):
    monkeypatch.setattr(app_mod, "installed_languages", lambda current: [current])

    async def scenario(app, fake, pilot):
        await pilot.press("l")
        [(message, severity)] = toasts(app)
        assert message.startswith("only Amharic is installed") and severity == "information"
        assert fake.languages == []

    drive(scenario)


def test_l_cycles_languages_through_the_controller(monkeypatch):
    two_languages(monkeypatch)

    async def scenario(app, fake, pilot):
        await pilot.press("l")
        await settle(pilot)
        assert fake.languages == ["swh"] and ("language: Swahili", "information") in toasts(app)
        await send(pilot, dataclasses.replace(READY, language="swh"))
        assert "Swahili" in screen_text(app)
        await pilot.press("l")
        await settle(pilot)
        assert fake.languages == ["swh", "amh"]

    drive(scenario)


def test_l_waits_while_the_model_loads(monkeypatch):
    two_languages(monkeypatch)

    async def scenario(app, fake, pilot):
        assert fake.started.wait(5)
        await send(pilot, ev.Loading("downloading model 40%"))
        await pilot.press("l")
        assert fake.languages == [] and toasts(app) == [("wait until the model has loaded", "warning")]
        fake.hold.set()
        await settle(pilot)

    drive(scenario, hold=True)


def test_l_degrades_when_the_controller_cannot_switch_language(monkeypatch):
    two_languages(monkeypatch)

    class Plain(FakeController):
        set_language = None

    async def scenario(app, fake, pilot):
        await pilot.press("l")
        assert toasts(app) == [("language switching is not available yet", "warning")]

    drive(scenario, factory=lambda c, e: Plain(c, e))


def test_l_reports_a_failed_switch(monkeypatch):
    two_languages(monkeypatch)

    async def scenario(app, fake, pilot):
        fake.set_language = lambda code: (_ for _ in ()).throw(RuntimeError("out of memory"))
        await pilot.press("l")
        await settle(pilot)
        assert ("RuntimeError: out of memory", "error") in toasts(app)

    drive(scenario)


def test_t_cycles_through_qal_light_and_the_builtin_themes_then_wraps():
    async def scenario(app, fake, pilot):
        assert app.theme == "qal"
        await pilot.press("t")
        assert app.theme == "qal-light"
        seen = [app.theme]
        for _ in range(len(app.available_themes) - 1):
            await pilot.press("t")
            seen.append(app.theme)
        assert app.theme == "qal" and len(set(seen)) == len(app.available_themes)

    drive(scenario)


def test_config_theme_is_honoured_and_a_typo_falls_back_with_a_toast():
    async def scenario(app, fake, pilot):
        assert app.theme == "nord"

    drive(scenario, cfg=Config(theme="nord"))

    async def typo(app, fake, pilot):
        assert app.theme == "qal" and toasts(app)[0][1] == "warning"

    drive(typo, cfg=Config(theme="nope"))


def test_q_stops_the_controller_and_exits():
    async def scenario(app, fake, pilot):
        await pilot.press("q")
        await settle(pilot)
        assert fake.stopped and not app.is_running

    drive(scenario)


def test_q_says_it_is_quitting_while_stop_waits():
    async def scenario(app, fake, pilot):
        release = threading.Event()
        fake.stop = lambda: release.wait(5)
        await pilot.press("q")
        assert "QUITTING" in pill(app) and "Quitting…" in screen_text(app)
        release.set()
        await settle(pilot)
        assert not app.is_running

    drive(scenario)


def test_q_still_exits_when_stop_fails_or_is_missing():
    class NoStop(FakeController):
        stop = None

    async def broken_stop(app, fake, pilot):
        fake.stop = lambda: 1 / 0
        await pilot.press("q")
        await settle(pilot)
        assert not app.is_running

    drive(broken_stop)

    async def no_stop(app, fake, pilot):
        await pilot.press("q")
        await settle(pilot)
        assert not app.is_running

    drive(no_stop, factory=lambda c, e: NoStop(c, e))


def test_footer_lists_every_binding():
    async def scenario(app, fake, pilot):
        footer = screen_text(app).splitlines()[-1]
        for hint in ("c Copy last", "l Language", "s Settings", "d Setup", "? Help", "q Quit"):
            assert hint in footer
        assert "Right Option dictate" in footer

    drive(scenario)


# -- compact mode ------------------------------------------------------------------------------------------


def lines_used(app) -> int:
    return len([line for line in screen_text(app).splitlines() if line.strip()])


def test_m_toggles_compact_mode():
    async def scenario(app, fake, pilot):
        assert not app.screen.has_class("-compact") and app.query_one("#history-panel").display
        await pilot.press("m")
        assert app.screen.has_class("-compact") and not app.query_one("#history-panel").display
        await send(pilot, ev.RecordingStarted("dictate"), ev.Level(0.4), ev.Partial("ሰላም"))
        text = screen_text(app)
        assert "ሰላም" in text and "REC" in text and lines_used(app) <= 5
        await pilot.press("m")
        assert not app.screen.has_class("-compact") and app.query_one("#history-panel").display

    drive(scenario)


@pytest.mark.parametrize("height", [3, 4, 5])
def test_compact_fits_three_to_five_lines(height):
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.Partial(SAMPLE))
        text = screen_text(app)
        assert "REC" in text and SAMPLE in text and ("q Quit" in text) == (height > 3)
        await send(pilot, ev.RecordingStopped("dictate", 1.0), done())
        text = screen_text(app)
        assert SAMPLE in text and "READY" in text and "ቃል" in text

    drive(scenario, size=(80, height), cfg=Config(compact=True))


def test_compact_is_a_strip_even_in_a_tall_window():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.Partial(SAMPLE))
        assert lines_used(app) <= 5

    drive(scenario, size=(80, 30), cfg=Config(compact=True))


def test_narrow_terminal_keeps_the_layout_readable():
    async def scenario(app, fake, pilot):
        await send(pilot, done("ሰላም ዓለም"))
        text = screen_text(app)
        assert "ሰላም ዓለም" in text and "Right Option dictate" not in text and "q Quit" in text
        assert all(len(line) <= 80 for line in text.splitlines())

    drive(scenario, size=(80, 24))


# -- thread safety and shutdown -----------------------------------------------------------------------------


def test_events_emitted_from_other_threads_reach_the_ui():
    async def scenario(app, fake, pilot):
        worker = threading.Thread(target=lambda: [fake.emit(ev.Notice(f"n{i}")) for i in range(3)])
        worker.start()
        worker.join()
        await pilot.pause()
        assert [m for m, _ in toasts(app)] == ["n0", "n1", "n2"]

    drive(scenario)


def test_run_hard_exits_only_while_the_controller_is_still_starting(monkeypatch):
    exits = []
    monkeypatch.setattr(app_mod.os, "_exit", exits.append)

    class Stub:
        download_thread, restart = None, False

        def __init__(self, cfg, **kwargs):
            self.boot_thread = types.SimpleNamespace(is_alive=lambda: still_loading)
            self.clean_exit = clean

        def run(self):
            pass

    monkeypatch.setattr(app_mod, "QalApp", Stub)
    still_loading, clean = True, True
    app_mod.run(Config())
    assert exits == [0]
    still_loading = False
    app_mod.run(Config())
    assert exits == [0]
    clean = False  # stop() gave up on a model call (a translation model download, say)
    app_mod.run(Config())
    assert exits == [0, 0]


def test_q_records_whether_stop_left_a_model_call_running():
    async def scenario(app, fake, pilot):
        fake.stop = lambda: False
        await pilot.press("q")
        await settle(pilot)
        assert not app.is_running and app.clean_exit is False

    drive(scenario)


def test_mic_name_follows_the_configured_device_and_never_raises(monkeypatch):
    asked = []

    def query_devices(device=None, kind=None):
        asked.append(device)
        if device == "missing":
            raise ValueError("no such device")
        return {"name": "USB Mic"}

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(query_devices=query_devices))
    assert app_mod.mic_name() == "USB Mic" and app_mod.mic_name("2") == "USB Mic"
    assert asked == [None, 2]  # "2" in a config file means device index 2
    assert app_mod.mic_name("missing") is None
