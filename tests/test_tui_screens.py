"""The setup, settings, key-capture and help screens, and the main-window changes that go with them."""
import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from test_tui import READY, SAMPLE, FakeController, done, drive, screen_text, send, settle, toasts

from qal_scribe import app as app_mod
from qal_scribe import doctor, models
from qal_scribe import events as ev
from qal_scribe.config import Config
from qal_scribe.doctor import Check
from qal_scribe.hotkey import INPUT_MONITORING_REASON
from qal_scribe.tui import screens
from qal_scribe.tui.screens import KeyCapture, SettingsScreen, SetupScreen

GRANTED = Check("Microphone permission", "ok", "granted")
NO_PASTE = Check("Accessibility", "fail", "Terminal cannot paste yet", "open it", "x-apple.systempreferences:paste")
NO_KEYS = Check("Input Monitoring", "fail", "the hotkeys cannot hear the keyboard", "open it", "x-apple:keys")


def config_file() -> Path:
    return Path(os.environ["XDG_CONFIG_HOME"]) / "qal-scribe" / "config.toml"


@pytest.fixture
def setup_env(monkeypatch):
    """Nothing cached, no permission rows; tests change `env.checks` / `env.cached`."""
    env = types.SimpleNamespace(cached=False, checks=[])
    monkeypatch.setattr(doctor, "is_cached", lambda *a: env.cached)
    monkeypatch.setattr(screens, "system_checks", lambda cfg: list(env.checks))
    monkeypatch.setattr(models, "missing_bytes", lambda *a: 2_900_000_000)
    return env


def shown(app, selector):
    return str(app.screen.query_one(selector).render())


# -- when the setup screen opens ---------------------------------------------------------------------------


def test_setup_opens_when_a_model_is_missing_and_boot_waits(setup_env):
    async def scenario(app, fake, pilot):
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen) and app.boot_thread is None and fake is None
        text = screen_text(app)
        assert "Amharic speech model" in text and "Translation to English" in text and "2.9 GB" in text
        await pilot.press("c")
        await settle(pilot)
        assert not isinstance(app.screen, SetupScreen) and app.boot_thread is not None

    drive(scenario, setup=None)


def test_setup_stays_closed_when_everything_is_in_place(setup_env):
    setup_env.cached = True

    async def scenario(app, fake, pilot):
        assert not isinstance(app.screen, SetupScreen) and fake is not None

    drive(scenario, setup=None)


@pytest.mark.parametrize(("platform", "check", "expected"), [
    ("darwin", Check("Accessibility", "fail", "x"), True),
    ("darwin", Check("Accessibility", "ok", "x"), False),
    ("linux", Check("Text insertion", "fail", "x"), False),  # the clipboard is a way to work
    ("linux", Check("Hotkeys", "warn", "x"), False),
    ("linux", Check("Microphone", "fail", "x"), True),
    ("darwin", Check("Microphone", "fail", "x"), True),
])
def test_setup_needed_rules(monkeypatch, setup_env, platform, check, expected):
    setup_env.cached = True
    setup_env.checks = [check]
    monkeypatch.setattr(sys, "platform", platform)
    assert screens.setup_needed(Config()) is expected


def test_setup_needed_for_a_missing_model_only_counts_what_the_config_uses(setup_env):
    assert [i.key for i in screens.items(Config())] == ["speech", "translation"]
    assert [i.key for i in screens.items(Config(translation=False))] == ["speech"]
    assert [i.key for i in screens.items(Config(translate_key=""))] == ["speech"]
    assert screens.setup_needed(Config(translation=False))  # the speech model is missing
    setup_env.cached = True
    assert not screens.setup_needed(Config())


def test_the_setup_subcommand_and_d_open_it_explicitly(setup_env):
    setup_env.cached = True

    async def scenario(app, fake, pilot):
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen) and "Translation to English" in screen_text(app)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SetupScreen)

    drive(scenario)


# -- permissions --------------------------------------------------------------------------------------------


def test_permission_rows_follow_the_system_without_a_restart_of_the_screen(setup_env):
    setup_env.checks = [GRANTED, NO_PASTE]

    async def scenario(app, fake, pilot):
        await pilot.pause()
        text = screen_text(app)
        assert "✗ Accessibility" in text and "Terminal cannot paste yet" in text and "Open Settings" in text
        setup_env.checks = [GRANTED, dataclasses.replace(NO_PASTE, status="ok", detail="granted")]
        app.screen.poll()
        await pilot.pause()
        assert "✓ Accessibility" in screen_text(app)

    drive(scenario, setup=True)


def test_the_button_asks_macos_first_then_opens_the_pane(setup_env, monkeypatch):
    setup_env.checks = [NO_PASTE]
    asked, opened = [], []
    monkeypatch.setattr(doctor, "request_permission", asked.append)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: opened.append(argv))

    async def scenario(app, fake, pilot):
        await pilot.pause()
        button = app.screen.query_one(".fix")
        button.press()
        await pilot.pause()
        assert asked == ["Accessibility"] and opened == []
        button.press()
        await pilot.pause()
        assert asked == ["Accessibility"] and opened == [["open", "x-apple.systempreferences:paste"]]

    drive(scenario, setup=True)


def test_input_monitoring_says_to_restart_the_terminal_once_asked(setup_env, monkeypatch):
    setup_env.checks = [NO_KEYS]
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.setattr(doctor, "request_permission", lambda name: None)

    async def scenario(app, fake, pilot):
        await pilot.pause()
        assert "restart iTerm2 to finish" not in screen_text(app)
        app.screen.query_one(".fix").press()
        await pilot.pause()
        assert "restart iTerm2 to finish" in screen_text(app)

    drive(scenario, setup=True)


def test_linux_rows_show_a_copyable_install_command(setup_env, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(doctor.platform, "freedesktop_os_release", lambda: {"ID": "fedora"})
    setup_env.checks = [Check("Microphone", "fail", "PortAudio library not found", "sudo dnf install portaudio")]

    async def scenario(app, fake, pilot):
        await pilot.pause()
        text = screen_text(app)
        assert "sudo dnf install portaudio" in text and "kernel-headers" in text
        await pilot.press("y")
        assert COPIED_LAST() == "sudo dnf install portaudio"

    from test_tui import COPIED

    def COPIED_LAST():
        return COPIED[-1]

    drive(scenario, setup=True)


# -- translation choice and downloads -----------------------------------------------------------------------


def test_unticking_translation_writes_the_config_and_drops_its_row(setup_env):
    async def scenario(app, fake, pilot):
        await pilot.pause()
        assert "Translation to English (nllb-600m)" in screen_text(app)
        app.screen.query_one("#tr-switch").toggle()
        await pilot.pause()
        assert app.cfg.translation is False and "translation = false" in config_file().read_text()
        assert "Translation to English (nllb-600m)" not in screen_text(app) and "Off: no translate key" in screen_text(app)

    drive(scenario, setup=True)


def test_choosing_a_model_is_saved_and_shows_its_licence(setup_env):
    async def scenario(app, fake, pilot):
        await pilot.pause()
        app.screen.query_one("#tr-model").value = "nllb-1.3b"
        await pilot.pause()
        assert app.cfg.translation_model == "nllb-1.3b" and 'translation_model = "nllb-1.3b"' in config_file().read_text()
        assert "non-commercial use only" in screen_text(app)

    drive(scenario, setup=True)


def test_progress_messages_move_the_bar_and_an_error_shows_retry(setup_env):
    async def scenario(app, fake, pilot):
        await pilot.pause()
        screen = app.screen
        screen.post_message(screens.DownloadProgress("speech", 1_450_000_000, 2_900_000_000))
        await pilot.pause()
        text = screen_text(app)
        assert "1.4 GB of 2.9 GB" in text and "━" in text
        screen.post_message(screens.DownloadFinished("speech", "could not reach the hub"))
        await pilot.pause()
        assert "✗ could not reach the hub" in screen_text(app) and str(screen.query_one("#download").label) == "Retry"

    drive(scenario, setup=True)


def test_download_runs_every_missing_model_then_closes_the_screen(setup_env, monkeypatch):
    fetched = []

    def fetch(repo, revision, patterns, progress):
        progress(50, 100)
        progress(100, 100)
        fetched.append(repo)
        setup_env.cached = len(fetched) == 2  # both models are cached after the second

    monkeypatch.setattr(models, "fetch", fetch)

    async def scenario(app, fake, pilot):
        await pilot.pause()
        app.screen.query_one("#download").press()
        await asyncio_wait(lambda: not isinstance(app.screen, SetupScreen), pilot)
        assert fetched == ["snapwre/hohe-asr-amharic", "facebook/nllb-200-distilled-600M"]
        await settle(pilot)
        assert app.boot_thread is not None

    drive(scenario, setup=True)


def test_a_failed_download_names_the_reason_and_keeps_the_screen(setup_env, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(*args):
        raise LocalEntryNotFoundError("cannot reach the Hub")

    monkeypatch.setattr(models, "fetch", offline)

    async def scenario(app, fake, pilot):
        await pilot.pause()
        app.screen.query_one("#download").press()
        await asyncio_wait(lambda: "Retry" in str(app.screen.query_one("#download").label), pilot)
        assert isinstance(app.screen, SetupScreen) and "cannot reach the Hub" in screen_text(app)

    drive(scenario, setup=True)


def test_sizes_that_cannot_be_looked_up_say_offline(setup_env, monkeypatch):
    def offline(*args):
        raise OSError("no network")

    monkeypatch.setattr(models, "missing_bytes", offline)

    async def scenario(app, fake, pilot):
        await asyncio_wait(lambda: "GB, offline" in screen_text(app), pilot)

    drive(scenario, setup=True)


async def asyncio_wait(condition, pilot, seconds=5.0):
    import asyncio

    for _ in range(int(seconds / 0.05)):
        await pilot.pause(0.05)
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("timed out waiting")


# -- settings -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["numbers_to_digits", "punctuation", "merge_homophones", "sounds", "preview"])
def test_live_switches_change_the_running_config_and_the_file(key):
    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        switch = app.screen.query_one(f"#set-{key}")
        was = getattr(app.cfg, key)
        switch.toggle()
        await pilot.pause()
        assert getattr(app.cfg, key) is (not was) and f"{key} = {str(not was).lower()}" in config_file().read_text()
        assert not app.screen.restart_needed

    drive(scenario)


def test_restart_settings_are_saved_and_offer_the_restart():
    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        assert "applies at restart" in screen_text(app) and app.screen.check_action("restart", ()) is False
        app.screen.query_one("#set-lm").toggle()
        await pilot.pause()
        assert "lm = false" in config_file().read_text() and app.screen.check_action("restart", ()) is True

    drive(scenario)


def test_theme_and_compact_apply_at_once():
    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one("#set-theme").value = "qal-light"
        app.screen.query_one("#set-compact").toggle()
        await pilot.pause()
        assert app.theme == "qal-light" and app.base.has_class("-compact")
        text = config_file().read_text()
        assert 'theme = "qal-light"' in text and "compact = true" in text

    drive(scenario)


def test_a_refused_value_shows_the_reason_leaves_the_file_alone_and_reverts_the_control(monkeypatch):
    config_file().parent.mkdir(parents=True)
    config_file().write_text("sounds = true\n")
    monkeypatch.setattr(doctor, "config_problem", lambda cfg: "sounds are not available here")

    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        switch = app.screen.query_one("#set-sounds")
        switch.toggle()
        await pilot.pause()
        assert "sounds are not available here" in screen_text(app)
        assert config_file().read_text() == "sounds = true\n" and app.cfg.sounds is True and switch.value is True

    drive(scenario)


def test_the_microphone_picker_lists_the_real_inputs(monkeypatch):
    monkeypatch.setattr(screens, "input_devices", lambda: ["MacBook Pro Microphone", "USB Mic"])

    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await asyncio_wait(lambda: len(list(app.screen.query_one("#set-mic")._options)) == 3, pilot)
        select = app.screen.query_one("#set-mic")
        select.value = "USB Mic"
        await pilot.pause()
        assert app.cfg.mic == "USB Mic" and 'mic = "USB Mic"' in config_file().read_text()
        select.value = ""
        await pilot.pause()
        assert app.cfg.mic is None and "mic =" not in config_file().read_text().replace("# mic =", "")

    drive(scenario)


def test_input_devices_skips_outputs(monkeypatch):
    devices = [{"name": "Mic", "max_input_channels": 1}, {"name": "Speakers", "max_input_channels": 0},
               {"name": "Mic", "max_input_channels": 2}]
    # a stand-in module: importing the real one fails where PortAudio has no server (CI)
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(query_devices=lambda: devices))
    assert screens.input_devices() == ["Mic"]


def test_t_remembers_the_theme_for_the_next_start():
    async def scenario(app, fake, pilot):
        await pilot.press("t")
        await pilot.pause()
        assert app.theme == "qal-light" and 'theme = "qal-light"' in config_file().read_text()

    drive(scenario)


def test_r_restarts_through_the_normal_quit_path():
    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one("#set-lm").toggle()
        await pilot.pause()
        await pilot.press("r")
        await settle(pilot)
        assert app.restart and not app.is_running

    drive(scenario)


def test_run_re_executes_the_same_command_after_a_restart(monkeypatch):
    calls = []
    monkeypatch.setattr(app_mod.os, "execv", lambda exe, argv: calls.append((exe, argv)))
    monkeypatch.setattr(sys, "argv", ["qal-scribe", "--config", "c.toml"])

    class Stub:
        boot_thread = download_thread = None
        clean_exit, restart = True, True

        def __init__(self, cfg, **kwargs):
            pass

        def run(self):
            pass

    monkeypatch.setattr(app_mod, "QalApp", Stub)
    app_mod.run(Config())
    assert calls == [(sys.executable, [sys.executable, "-m", "qal_scribe", "--config", "c.toml"])]


# -- key capture --------------------------------------------------------------------------------------------


class Listener:
    stopped = False

    def stop(self):
        self.stopped = True


def open_capture(monkeypatch, listener=Listener):
    monkeypatch.setattr(screens, "listen", lambda on_press: listener() if listener else None)


def test_capture_accepts_a_modifier_or_function_key_and_saves_it(monkeypatch):
    open_capture(monkeypatch)

    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one("#set-dictate_key").press()
        await pilot.pause()
        capture = app.screen
        assert isinstance(capture, KeyCapture) and "Press the key you want for Dictate" in screen_text(app)
        capture.post_message(KeyCapture.Heard("f13"))
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen) and app.cfg.dictate_key == "f13"
        assert 'dictate_key = "f13"' in config_file().read_text() and "F13" in screen_text(app)
        assert capture.listener is None

    drive(scenario)


def test_capture_rejects_character_keys_and_the_other_modes_key(monkeypatch):
    open_capture(monkeypatch)

    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one("#set-dictate_key").press()
        await pilot.pause()
        app.screen.post_message(KeyCapture.Heard(None))
        await pilot.pause()
        assert "pick a key you never type with" in screen_text(app) and isinstance(app.screen, KeyCapture)
        app.screen.post_message(KeyCapture.Heard(app.cfg.translate_key))
        await pilot.pause()
        assert "is the other mode's key" in screen_text(app) and isinstance(app.screen, KeyCapture)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen) and app.cfg.dictate_key == "alt_r"

    drive(scenario)


def test_capture_converts_pynput_keys_and_ignores_typing_keys():
    pynput = pytest.importorskip("pynput.keyboard", exc_type=ImportError)  # no display on CI
    seen = []
    capture = KeyCapture("Dictate", None, False)
    capture.post_message = seen.append
    capture.heard(pynput.Key.f13)
    capture.heard(pynput.KeyCode.from_char("a"))
    assert [m.name for m in seen] == ["f13", None]


def test_capture_falls_back_to_a_list_where_global_keys_cannot_work(monkeypatch):
    open_capture(monkeypatch, listener=None)  # Wayland, no display

    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one("#set-translate_key").press()
        await pilot.pause()
        select = app.screen.query_one("#capture-list")
        assert select.display and "Off" in [str(label) for label, _ in select._options]
        select.value = ""  # Off
        await pilot.pause()
        assert app.cfg.translate_key == "" and 'translate_key = ""' in config_file().read_text()

    drive(scenario)


def test_the_hotkeys_are_paused_while_a_key_is_captured(monkeypatch):
    open_capture(monkeypatch)

    class Hotkeys:
        paused = False

    async def scenario(app, fake, pilot):
        fake.hotkeys = Hotkeys()
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one("#set-dictate_key").press()
        await pilot.pause()
        assert fake.hotkeys.paused
        await pilot.press("escape")
        await pilot.pause()
        assert not fake.hotkeys.paused

    drive(scenario)


# -- main window ----------------------------------------------------------------------------------------------


class Failing(FakeController):
    def __init__(self, cfg, emit, error=""):
        super().__init__(cfg, emit)
        self.error = error

    def start(self):
        raise ValueError(self.error)


@pytest.mark.parametrize(("error", "fix"), [
    ("cannot open mic 'Blue Yeti': no input device", "press s to pick a microphone"),
    ("qal-scribe is already running (control socket /x is in use)", "another qal-scribe is running: quit it first"),
    ("something else", "run `qal-scribe doctor`"),
])
def test_a_failed_start_has_one_fix_line_and_no_toast(error, fix):
    async def scenario(app, fake, pilot):
        await settle(pilot)
        text = screen_text(app)
        assert fix in text and toasts(app) == [] and "s Settings · d Setup · q Quit" in text

    drive(scenario, factory=lambda c, e: Failing(c, e, error), size=(100, 30))


def test_dead_keys_are_dimmed_not_offered():
    async def scenario(app, fake, pilot):
        assert app.check_action("copy", ()) is None and app.check_action("copy_source", ()) is False
        await send(pilot, done())
        assert app.check_action("copy", ()) is True
        await send(pilot, done("hello", mode="translate"))
        assert app.check_action("copy_source", ()) is True

    drive(scenario)


def test_keys_that_do_nothing_after_a_failed_start_are_dimmed():
    async def scenario(app, fake, pilot):
        await settle(pilot)
        assert app.check_action("language", ()) is None and app.check_action("compact", ()) is None
        assert app.check_action("settings", ()) is True

    drive(scenario, factory=lambda c, e: Failing(c, e, "x"))


def test_main_window_keys_are_off_while_another_screen_is_up():
    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        assert app.check_action("settings", ()) is False and app.check_action("quit", ()) is True

    drive(scenario)


class MacPermissions(FakeController):
    def start(self):
        self.started.set()
        self.emit(ev.Notice("no Accessibility permission: add your terminal app under System Settings", "warning"))
        self.emit(ev.Notice(INPUT_MONITORING_REASON, "warning"))
        self.emit(READY)


def test_setup_problems_are_one_line_callouts_with_the_way_to_fix_them():
    async def scenario(app, fake, pilot):
        text = screen_text(app)
        assert "⚠ Accessibility is off: cannot paste · press d" in text
        assert "⚠ Input Monitoring is off: hotkeys cannot hear the keyboard · press d" in text
        assert "Hotkeys are off until Input Monitoring is on." in text and "Hold" not in text
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen)

    drive(scenario, factory=lambda c, e: MacPermissions(c, e), size=(100, 30))


def test_compact_mode_summarises_the_setup_problems_in_one_line():
    async def scenario(app, fake, pilot):
        text = screen_text(app)
        assert "⚠ 2 setup problems · press d" in text and "Accessibility is off" not in text

    drive(scenario, factory=lambda c, e: MacPermissions(c, e), size=(80, 5), cfg=Config(compact=True))


def test_the_hero_names_the_current_language_and_platform_keys(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    async def scenario(app, fake, pilot):
        assert "to dictate Amharic" in screen_text(app)
        await send(pilot, dataclasses.replace(READY, language="tir", hotkeys={"Right Alt": "dictate"}))
        text = screen_text(app)
        assert "to dictate Tigrinya" in text and "Amharic" not in text.split("Loading")[0].split("READY")[1]

    drive(scenario)


def test_a_short_terminal_keeps_rows_for_the_history():
    async def scenario(app, fake, pilot):
        await send(pilot, done("አንድ"), done("ሁለት"), done("ሦስት"))
        assert app.base.has_class("-short") and not app.base.has_class("-tall")
        text = screen_text(app)
        assert "አንድ" in text and "ሁለት" in text

    drive(scenario, size=(80, 14))
    drive(lambda app, fake, pilot: _tall(app), size=(80, 24))


async def _tall(app):
    assert app.base.has_class("-tall")


def test_highlighting_a_history_row_shows_its_full_text_and_escape_goes_back():
    long = " ".join([SAMPLE] * 6)

    async def scenario(app, fake, pilot):
        await send(pilot, done(long), done("አዲስ"))
        await pilot.pause(0.5)  # the list re-highlights as rows shift: that is not the user browsing
        assert app.query_one("#stage").border_title == "Dictate"
        await pilot.press("down", "down")  # the first highlights the newest, which the stage already shows
        await pilot.pause()
        assert app.query_one("#stage").border_title.startswith("History · ")
        assert screen_text(app).count(SAMPLE) >= 3  # the whole text, not two clipped lines
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#stage").border_title == "Dictate" and "አዲስ" in shown_stage(app)
        await send(pilot, done("ሦስተኛ"))
        assert "ሦስተኛ" in shown_stage(app)

    drive(scenario, size=(120, 40))


def shown_stage(app):
    return str(app.query_one("#transcript").render())


def test_a_copies_the_amharic_source_of_a_translated_row():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.Transcribed("translate", "hello world", SAMPLE, 2.0, 0.2, True))
        await pilot.press("a")
        from test_tui import COPIED

        assert COPIED[-1] == SAMPLE

    drive(scenario)


def test_help_lists_the_configured_keys_and_any_key_closes_it():
    async def scenario(app, fake, pilot):
        await pilot.press("question_mark")
        await pilot.pause()
        text = screen_text(app)
        assert "Hold Right Option to dictate Amharic" in text and "Settings" in text and "qal-scribe doctor" in text
        await pilot.press("x")
        await pilot.pause()
        assert not isinstance(app.screen, screens.HelpScreen)

    drive(scenario)


def test_the_status_line_says_whether_translation_is_on():
    async def scenario(app, fake, pilot):
        assert "→EN nllb-1.3b" in shown_status(app)

    drive(scenario, cfg=Config(translation_model="nllb-1.3b"))

    async def off(app, fake, pilot):
        assert "translation off" in shown_status(app)

    drive(off, cfg=Config(translation=False))


def shown_status(app):
    return str(app.query_one("#status").render())


def test_the_key_reminder_is_readable_in_the_light_theme():
    css = (Path(app_mod.__file__).parent / "tui" / "qal.tcss").read_text()
    keys = css[css.index("#keys {"):].split("}")[0]
    assert "$text-muted" in keys and "$text-disabled" not in keys


# -- review fixes --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["space", "enter", "tab", "backspace", "delete", "left", "up"])
def test_capture_refuses_keys_you_type_with(name):
    async def scenario(app, fake, pilot):
        app.push_screen(KeyCapture("Dictate", None, False))
        await pilot.pause()
        app.screen.post_message(KeyCapture.Heard(name))
        await pilot.pause()
        assert isinstance(app.screen, KeyCapture) and "never type with" in screen_text(app)
        app.screen.post_message(KeyCapture.Heard("f13"))
        await pilot.pause()
        assert not isinstance(app.screen, KeyCapture)

    drive(scenario)


def test_the_short_stage_shows_the_result_and_the_live_draft():
    async def scenario(app, fake, pilot):
        await send(pilot, ev.RecordingStarted("dictate"), ev.Partial("ቀጥታ ረቂቅ ጽሑፍ"))
        assert "ቀጥታ ረቂቅ ጽሑፍ" in screen_text(app)
        await send(pilot, ev.RecordingStopped("dictate", 3.2), done("የመጨረሻ ውጤት"))
        assert "የመጨረሻ ውጤት" in screen_text(app).split("History")[0]

    drive(scenario, size=(80, 14))


def test_continue_is_off_while_a_download_runs(setup_env, monkeypatch):
    monkeypatch.setattr(SetupScreen, "start_download", lambda self: setattr(self, "running", True) or self.show_items())

    async def scenario(app, fake, pilot):
        await pilot.pause()
        app.screen.query_one("#download").press()
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen) and app.boot_thread is None
        assert app.screen.query_one("#continue").disabled

    drive(scenario, setup=True)


@pytest.mark.parametrize(("platform", "title"), [("darwin", "Permissions"), ("linux", "Your system")])
def test_setup_fits_80x24_with_the_permissions_title_in_view(setup_env, monkeypatch, platform, title):
    monkeypatch.setattr(sys, "platform", platform)
    setup_env.checks = [NO_PASTE]

    async def scenario(app, fake, pilot):
        await pilot.pause()
        text = screen_text(app)
        assert title in text and "Download" in text and "Continue" in text

    drive(scenario, setup=True, size=(80, 24))


def test_generic_callouts_point_at_setup():
    assert app_mod.summarize("text insertion needs wtype").endswith("· press d")


def test_switching_a_restart_setting_back_withdraws_the_restart_offer():
    async def scenario(app, fake, pilot):
        await pilot.press("s")
        await pilot.pause()
        screen = app.screen
        switch = screen.query_one("#set-lm")
        switch.toggle()
        await pilot.pause()
        assert screen.restart_needed
        switch.toggle()
        await pilot.pause()
        assert not screen.restart_needed

    drive(scenario)


def test_status_line_says_translation_off_without_a_translate_key():
    async def scenario(app, fake, pilot):
        assert "translation off" in screen_text(app) and "→EN" not in screen_text(app)

    drive(scenario, cfg=Config(translate_key=""), size=(100, 24))
