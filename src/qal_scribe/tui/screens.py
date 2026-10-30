"""The screens pushed over the main window: first-run setup, settings, key capture and help.

Contract: setup_needed(cfg) says whether QalApp should open SetupScreen at start. Screens read and write the app's
shared `cfg` and `config_path`; save_changes(app, changes) writes config.toml first and only then changes `cfg`, and
returns the problem text instead of raising. Downloads and the size look-ups run in daemon threads (a Textual worker
would be joined at exit and hold `q` hostage) and report through messages. Importing this module must stay torch-free.
"""
import importlib.util
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Select, Static, Switch

from qal_scribe import config, doctor, models
from qal_scribe.doctor import Check
from qal_scribe.hotkey import is_wayland, key_label
from qal_scribe.translate import TRANSLATION_MODELS
from qal_scribe.tui.widgets import TopBar, progress_bar

MARKS = {"ok": ("✓", "$success"), "warn": ("!", "$warning"), "fail": ("✗", "$error")}
COMMON_KEYS = ["alt_r", "ctrl_r", "cmd_r", "shift_r", "alt_l", *(f"f{n}" for n in range(13, 21))]


def background(fn, *args) -> threading.Thread:
    thread = threading.Thread(target=fn, args=args, daemon=True)
    thread.start()
    return thread


def gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f} GB"


# -- what to download and what to check ---------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One download: a model the current config needs."""

    key: str
    title: str
    repo: str
    revision: str
    patterns: tuple[str, ...]
    weights: tuple[str, ...]  # cached when any of these is in the cache
    estimate_gb: float
    license: str
    extra: str | None = None  # a file that must be cached too (the language model)

    def cached(self) -> bool:
        has = doctor.is_cached
        return has(self.repo, self.revision, *self.weights) and (not self.extra or has(self.repo, self.revision, self.extra))


def items(cfg: config.Config) -> list[Item]:
    spec = models.get(cfg.language)
    lm = bool(cfg.lm and spec.lm_file and importlib.util.find_spec("flashlight"))
    out = [Item("speech", f"{spec.language_name} speech model", spec.repo, spec.revision,
                tuple(models.patterns_for(spec, lm)), doctor.WEIGHT_FILES, 2.4 + 0.5 * lm, spec.license,
                spec.lm_file if lm else None)]
    if cfg.translation and cfg.translate_key:
        t = TRANSLATION_MODELS[cfg.translation_model]
        out.append(Item("translation", f"Translation to English ({cfg.translation_model})", t.repo, t.revision,
                        t.patterns, (t.weights,), t.size_gb, t.license))
    return out


def system_checks(cfg: config.Config) -> list[Check]:
    """The rows of the setup screen: permissions on a Mac (plus the microphone when it fails), the system
    pieces on Linux."""
    mic = doctor.check_microphone(cfg.mic)
    if sys.platform != "darwin":
        return [c for c in (mic, doctor.check_paste(), doctor.check_hotkeys()) if c]
    rows = [c for c in (doctor.check_microphone_permission(), doctor.check_paste(), doctor.check_input_monitoring()) if c]
    for i, c in enumerate(rows):
        if c.name == "Microphone permission" and c.status == "ok" and doctor.microphone_permission() == 0:
            rows[i] = Check(c.name, "warn", "not asked yet", c.fix, doctor.PANES["Microphone"])
    return rows + ([mic] if mic.status == "fail" else [])


def setup_needed(cfg: config.Config) -> bool:
    """A model is missing, or (macOS) a permission, or (anywhere) the microphone. Linux text insertion and
    hotkey problems do not count: the clipboard and `qal-scribe toggle` are ways to work."""
    if any(not i.cached() for i in items(cfg)):
        return True
    return any(c.status == "fail" and (sys.platform == "darwin" or c.name == "Microphone") for c in system_checks(cfg))


def save_changes(app, changes: dict[str, object | None]) -> str | None:
    """Write `changes` to config.toml, then to the running config. The problem text when it was refused."""
    try:
        config.save(app.config_path, changes)
    except (ValueError, OSError) as e:
        return str(e)
    for key, value in changes.items():
        setattr(app.cfg, key, value)
    return None


# -- setup ------------------------------------------------------------------------------------------------


class DownloadProgress(Message):
    def __init__(self, key: str, done: int, total: int) -> None:
        super().__init__()
        self.key, self.done, self.total = key, done, total


class DownloadFinished(Message):
    def __init__(self, key: str, error: str | None) -> None:
        super().__init__()
        self.key, self.error = key, error


class DownloadsEnded(Message):
    pass


class SizeKnown(Message):
    def __init__(self, key: str, nbytes: int | None) -> None:
        super().__init__()
        self.key, self.nbytes = key, nbytes


class CheckRow(Vertical):
    """One permission / system check: mark, name, what is wrong, and a button when there is something to open."""

    def __init__(self, check: Check) -> None:
        super().__init__(classes="check")
        self.check = check

    def compose(self) -> ComposeResult:
        with Horizontal(classes="line"):
            yield Static("", classes="mark")
            yield Static(self.check.name, classes="name")
            yield Static("", classes="detail")
            yield Button("Open Settings", name=self.check.name, classes="fix", compact=True)
        yield Static("", classes="command")  # Linux: the install command, selectable

    def show(self, check: Check, detail: str, command: str = "") -> None:
        self.check = check
        mark, colour = MARKS[check.status]
        self.query_one(".mark", Static).update(Content.styled(mark, f"bold {colour}"))
        self.query_one(".detail", Static).update(detail)
        self.query_one(".fix", Button).display = check.status != "ok" and bool(check.url)
        self.query_one(".command", Static).update(command)
        self.query_one(".command").display = bool(command)


class SetupScreen(Screen[bool]):
    """Permissions, then the models: choose translation, see the size, download with a bar per model."""

    AUTO_FOCUS = "#download:enabled, #continue"
    BINDINGS = [  # noqa: RUF012 (Textual convention)
        Binding("enter", "download", "Download"),
        Binding("c,escape", "continue", "Continue"),
        Binding("y", "copy_command", "Copy command"),
    ]
    DEFAULT_CSS = """
    SetupScreen { background: $background; }
    SetupScreen #body { align-horizontal: center; }
    SetupScreen #setup { width: 76; max-width: 100%; height: auto; padding: 0 1; }
    SetupScreen .panel {
        height: auto; margin-top: 1; padding: 0 1; background: $surface; border: round $border;
        border-title-color: $text-muted; border-title-style: bold;
    }
    SetupScreen .check, SetupScreen .line { height: auto; }
    SetupScreen .check .mark { width: 2; }
    SetupScreen .check .name { width: 24; }
    SetupScreen .check .detail { width: 1fr; color: $text-muted; }
    SetupScreen .check .fix { margin-left: 1; }
    SetupScreen #checks, SetupScreen #items { height: auto; }
    SetupScreen .command { padding-left: 2; color: $accent; }
    SetupScreen .note { color: $text-muted; }
    SetupScreen Switch { border: none; padding: 0 1; }
    SetupScreen #tr-row { height: 1; }
    SetupScreen #tr-info { color: $text-muted; }
    SetupScreen .item { height: auto; margin-top: 1; }
    SetupScreen #actions { height: 3; align-horizontal: right; margin-top: 1; }
    SetupScreen #actions Button { margin-left: 1; }
    SetupScreen #problem { color: $error; }
    """

    def __init__(self, cfg: config.Config, explicit: bool = False) -> None:
        super().__init__()
        self.cfg, self.explicit = cfg, explicit
        self.todo = items(cfg)
        self.ask = explicit or any(not i.cached() for i in self.todo)  # the translation question
        self.sizes: dict[str, int | None | str] = {}  # bytes, None = offline, absent = still asking
        self.progress: dict[str, tuple[int, int]] = {}
        self.errors: dict[str, str] = {}
        self.asked: set[str] = set()  # permissions the system request was already made for
        self.running = False
        self.checks: list[Check] = []

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with VerticalScroll(id="body"), Vertical(id="setup"):
            with Vertical(id="system", classes="panel"):
                yield Vertical(id="checks")
                yield Static("", id="restart-hint", classes="note")
            with Vertical(id="models", classes="panel"):
                if self.ask:
                    with Horizontal(id="tr-row"):
                        yield Switch(self.cfg.translation, id="tr-switch")
                        yield Static("Translation to English")
                    yield Select([(f"{k}  ·  {t.size_gb:g} GB", k)
                                  for k, t in TRANSLATION_MODELS.items()], value=self.cfg.translation_model,
                                 allow_blank=False, id="tr-model", compact=True)
                    yield Static("", id="tr-info")
                with Vertical(id="items"):
                    for key in ("speech", "translation"):  # hidden while the config does not use that model
                        yield Static("", id=f"item-{key}", classes="item")
                yield Static("", id="problem")
                with Horizontal(id="actions"):
                    yield Button("Download", id="download", variant="primary")
                    yield Button("Continue", id="continue")
        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        top = self.query_one(TopBar)
        top.set_status(Content("Setup"))
        top.set_pill("SETUP")
        self.query_one("#system").border_title = "Permissions" if sys.platform == "darwin" else "Your system"
        self.query_one("#models").border_title = "Models"
        self.poll()
        self.set_interval(1, self.poll)
        self.show_items()
        self.measure()

    # -- permissions / system ---------------------------------------------------------------------------

    def poll(self) -> None:
        checks = system_checks(self.cfg)
        box = self.query_one("#checks")
        if [c.name for c in checks] != [c.name for c in self.checks]:  # first time, or the microphone row came or went
            box.remove_children()
            for c in checks:
                box.mount(CheckRow(c))
            self.checks = checks
            self.call_after_refresh(self.show_checks)
        else:
            self.checks = checks
            self.show_checks()

    def show_checks(self) -> None:
        terminal = doctor.terminal_name()
        for row, c in zip(self.query(CheckRow), self.checks, strict=False):
            detail = c.detail
            if c.name == "Input Monitoring" and c.status != "ok" and c.name in self.asked:
                detail = f"restart {terminal} to finish"
            row.show(c, detail, c.fix if c.status != "ok" and sys.platform != "darwin" else "")
        failing = any(c.status != "ok" for c in self.checks)
        if sys.platform == "darwin":
            hint = f"After turning one on, quit {terminal} (⌘Q) and reopen it." if failing else ""
        else:
            hint = ("Reinstalling needs build tools for the keyboard library: " + doctor.linux_install("build")
                    if failing else "")
        self.query_one("#restart-hint", Static).update(hint)
        self.query_one("#system").display = bool(self.checks)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "download":
                self.start_download()
            case "continue":
                self.action_continue()
            case _:
                if check := next((c for c in self.checks if c.name == event.button.name), None):
                    self.fix(check)

    def fix(self, check: Check) -> None:
        """First press asks macOS for the permission (that also lists the terminal); later presses open the pane."""
        denied = check.name == "Microphone permission" and check.status == "fail"  # macOS asks only once
        if check.name not in self.asked and not denied:
            self.asked.add(check.name)
            doctor.request_permission(check.name.removesuffix(" permission"))
        else:
            subprocess.run(["open", check.url], check=False)
        self.show_checks()

    def action_copy_command(self) -> None:
        commands = [c.fix for c in self.checks if c.status != "ok" and c.fix and sys.platform != "darwin"]
        if commands:
            self.app.copy(commands[0])

    # -- models -----------------------------------------------------------------------------------------

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.change({"translation": event.value}, event.switch)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "tr-model" and event.value != self.cfg.translation_model:
            self.change({"translation_model": event.value}, event.select)

    def change(self, changes: dict, control) -> None:
        problem = save_changes(self.app, changes)
        self.query_one("#problem", Static).update(problem or "")
        if problem:
            with self.prevent(Switch.Changed, Select.Changed):
                control.value = getattr(self.cfg, next(iter(changes)))
            return
        self.todo = items(self.cfg)
        self.show_items()
        self.measure()

    def show_items(self) -> None:
        for row in self.query(".item"):
            item = next((i for i in self.todo if f"item-{i.key}" == row.id), None)
            row.display = item is not None
            if item:
                row.update(self.item_line(item))
        if self.ask:
            t = TRANSLATION_MODELS[self.cfg.translation_model]
            terms = "non-commercial use only" if "NC" in t.license else "free to use commercially"
            self.query_one("#tr-info", Static).update(terms if self.cfg.translation else
                                                      "Off: no translate key, nothing to download")
        todo = [i for i in self.todo if not i.cached()]
        download = self.query_one("#download", Button)
        download.disabled = self.running or not todo
        download.label = "Retry" if self.errors and todo else "Download"
        self.query_one("#continue", Button).disabled = self.running  # leaving would start the model load beside the download
        self.refresh_bindings()

    def item_line(self, item: Item) -> Content:
        head = Content.assemble((item.title, "bold"), f"  {item.license}")
        if item.key in self.errors:
            return Content.assemble(head, "\n", (f"✗ {self.errors[item.key]}", "$error"))
        if item.cached():
            return Content.assemble(head, "\n", ("✓ downloaded", "$success"))
        size = self.sizes.get(item.key, "…")
        about = f"about {item.estimate_gb:g} GB"
        size = f"{about}, offline" if size is None else about if size == "…" else gb(size)
        if item.key in self.progress:
            done, total = self.progress[item.key]
            filled, rest = progress_bar(round(100 * done / total) if total else 0, 24)
            return Content.assemble(head, "\n", (filled, "$accent"), (rest, "$border"),
                                    f"  {gb(done)} of {gb(total)}")
        return Content.assemble(head, "\n", (size, "$text-muted"))

    def measure(self) -> None:
        for item in self.todo:
            if not item.cached() and item.key not in self.sizes:
                self.sizes[item.key] = "…"
                background(self.measure_one, item)

    def measure_one(self, item: Item) -> None:
        try:
            nbytes = models.missing_bytes(item.repo, item.revision, list(item.patterns))
        except Exception:  # noqa: BLE001 - offline or the hub is down: the size just stays unknown
            nbytes = None
        self.post_message(SizeKnown(item.key, nbytes))

    def on_size_known(self, message: SizeKnown) -> None:
        self.sizes[message.key] = message.nbytes
        self.show_items()

    def start_download(self) -> None:
        todo = [i for i in self.todo if not i.cached()]
        if self.running or not todo:
            return
        self.running, self.errors = True, {}
        self.app.download_thread = background(self.download, todo)
        self.show_items()

    def download(self, todo: list[Item]) -> None:
        for item in todo:
            last = 0.0

            def report(done: int, total: int, key: str = item.key) -> None:
                nonlocal last
                if time.monotonic() - last > 0.1 or done >= total:
                    last = time.monotonic()
                    self.post_message(DownloadProgress(key, done, total))

            try:
                models.fetch(item.repo, item.revision, list(item.patterns), report)
            except Exception as e:  # noqa: BLE001 - whatever stopped it goes on the row, with a Retry
                self.post_message(DownloadFinished(item.key, models.download_problem(e) or f"{type(e).__name__}: {e}"))
            else:
                self.post_message(DownloadFinished(item.key, None))
        self.post_message(DownloadsEnded())

    def on_download_progress(self, message: DownloadProgress) -> None:
        self.progress[message.key] = (message.done, message.total)
        self.show_items()

    def on_download_finished(self, message: DownloadFinished) -> None:
        self.progress.pop(message.key, None)
        if message.error:
            self.errors[message.key] = message.error
        self.show_items()

    def on_downloads_ended(self, message: DownloadsEnded) -> None:
        self.running = False
        self.show_items()
        if not self.errors and all(i.cached() for i in self.todo):
            self.dismiss(True)

    def action_continue(self) -> None:
        if not self.running:
            self.dismiss(True)

    def action_download(self) -> None:
        self.start_download()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        match action:
            case "copy_command":
                return sys.platform != "darwin"
            case "continue":
                return None if self.running else True
            case "download":
                return not self.running and any(not i.cached() for i in self.todo)
        return True


# -- settings ---------------------------------------------------------------------------------------------


class KeyCapture(ModalScreen[str | None]):
    """Wait for one key press (pynput, since a terminal never reports Right Option on its own). Dismisses with
    the pynput key name, "" for Off, None when cancelled."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]  # noqa: RUF012 (Textual convention)
    DEFAULT_CSS = """
    KeyCapture { align: center middle; }
    KeyCapture #capture { width: 62; max-width: 100%; height: auto; padding: 1 2; background: $surface;
                          border: round $accent; }
    KeyCapture #capture-error { color: $error; }
    KeyCapture Select, KeyCapture Button { margin-top: 1; }
    """

    TYPING = frozenset({"space", "enter", "tab", "backspace", "delete", "up", "down", "left", "right"})

    class Heard(Message):
        def __init__(self, name: str | None) -> None:
            super().__init__()
            self.name = name  # None: a key that types a character

    def __init__(self, label: str, other: str | None, allow_off: bool) -> None:
        super().__init__()
        self.label, self.other, self.allow_off = label, other, allow_off
        self.listener = None

    def compose(self) -> ComposeResult:
        options = [(key_label(k), k) for k in COMMON_KEYS if k != self.other]
        if self.allow_off:
            options.append(("Off", ""))
        with Vertical(id="capture"):
            yield Static(f"Press the key you want for {self.label}. Esc cancels.", id="capture-prompt")
            yield Static("", id="capture-error")
            yield Select(options, prompt="Choose a key", id="capture-list", compact=True)
            yield Button("Choose from a list instead", id="capture-fallback", compact=True)

    def on_mount(self) -> None:
        self.app.set_keys_paused(True)
        self.listener = listen(self.heard)
        self.query_one("#capture-list").display = self.listener is None
        self.query_one("#capture-fallback").display = self.listener is not None

    def heard(self, key) -> None:
        """pynput's thread."""
        from pynput import keyboard

        self.post_message(self.Heard(key.name if isinstance(key, keyboard.Key) else None))

    def on_key_capture_heard(self, message: Heard) -> None:
        if message.name is None or message.name in self.TYPING:
            self.query_one("#capture-error", Static).update(
                "pick a key you never type with, such as Right Option or F13")
        elif message.name == "esc":
            self.finish(None)
        elif message.name == self.other:
            self.query_one("#capture-error", Static).update(f"{key_label(message.name)} is the other mode's key")
        else:
            self.finish(message.name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.stop_listening()
        event.button.display = False
        self.query_one("#capture-list").display = True

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self.finish(str(event.value))

    def stop_listening(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def finish(self, name: str | None) -> None:
        self.stop_listening()
        self.app.set_keys_paused(False)
        self.dismiss(name)

    def action_cancel(self) -> None:
        self.finish(None)


def listen(on_press):
    """A started pynput listener, or None where global keys cannot work (Wayland, no display, no permission)."""
    if is_wayland():
        return None
    try:
        from pynput import keyboard

        listener = keyboard.Listener(on_press=on_press, daemon=True)
        listener.start()
    except Exception:  # noqa: BLE001 - no X display, missing library: the caller shows a list instead
        return None
    return listener


def input_devices() -> list[str]:
    try:
        import sounddevice

        names = [d["name"] for d in sounddevice.query_devices() if d["max_input_channels"] > 0]
    except Exception:  # noqa: BLE001 - no audio stack: the picker keeps the current value only
        return []
    return list(dict.fromkeys(names))


class MicsLoaded(Message):
    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = names


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    help: str
    kind: str  # switch | select | mic | key
    live: bool


SETTINGS = [
    Setting("translation", "Translation to English", "A second hotkey that speaks English text", "switch", False),
    Setting("translation_model", "Translation model", "Bigger models are slower and download more", "select", False),
    Setting("lm", "Language model", "A little more accurate; needs the flashlight-text decoder", "switch", False),
    Setting("numbers_to_digits", "Numbers as digits", "Write spoken numbers as 1 2 3", "switch", True),
    Setting("punctuation", "Punctuation (experimental)", "Add punctuation with simple rules", "switch", True),
    Setting("merge_homophones", "Merge homophones", "Spell sound-alike letters one way", "switch", True),
    Setting("sounds", "Sounds", "Start, stop and error sounds (v mutes)", "switch", True),
    Setting("preview", "Live preview", "A draft of the text while you are still speaking", "switch", True),
    Setting("mic", "Microphone", "Which input to record from", "mic", False),
    Setting("dictate_key", "Dictate key", "Hold to record, release to paste", "key", False),
    Setting("translate_key", "Translate key", "Hold to record and paste English", "key", False),
    Setting("theme", "Theme", "Colours of this window", "select", True),
    Setting("compact", "Compact", "A small strip for an always-visible window", "switch", True),
    Setting("check_updates", "Check for updates", "Once a day, ask PyPI for a newer version", "switch", False),
]


class SettingsScreen(Screen):
    """Every control saves at once. Live settings apply now; the others say "applies at restart"."""

    BINDINGS = [  # noqa: RUF012 (Textual convention)
        Binding("escape", "close", "Close"),
        Binding("r", "restart", "Restart now"),
    ]
    DEFAULT_CSS = """
    SettingsScreen { background: $background; }
    SettingsScreen #rows { align-horizontal: center; }
    SettingsScreen #settings { width: 76; max-width: 100%; height: auto; padding: 0 1; }
    SettingsScreen .row { height: auto; margin-top: 1; }
    SettingsScreen Switch { border: none; padding: 0 1; }
    SettingsScreen .text { width: 1fr; height: auto; }
    SettingsScreen .help { color: $text-muted; }
    SettingsScreen .badge { color: $warning; }
    SettingsScreen .row > Select, SettingsScreen .row > Button { width: 30; }
    SettingsScreen #problem { color: $error; padding: 0 1; height: auto; }
    """

    def __init__(self, cfg: config.Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.restart_needed = False
        self.original: dict[str, object] = {}  # restart settings as they were when the screen opened

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with VerticalScroll(id="rows"), Vertical(id="settings"):
            yield Static("", id="problem")
            for s in SETTINGS:
                with Horizontal(classes="row"):
                    with Vertical(classes="text"):
                        yield Static(Content.styled(s.label, "bold"))
                        yield Static(self.help_line(s), classes="help", id=f"help-{s.key}")
                        if not s.live:
                            yield Static("applies at restart", classes="badge")
                    yield self.control(s)
        yield Footer(show_command_palette=False)

    def help_line(self, s: Setting) -> Content:
        text = s.help
        if s.key == "translation" and self.cfg.translation:
            t = TRANSLATION_MODELS[self.cfg.translation_model]
            if not doctor.is_cached(t.repo, t.revision, t.weights):
                text = f"downloads {t.size_gb:g} GB at restart"
        return Content(text)

    def control(self, s: Setting):
        value = getattr(self.cfg, s.key)
        match s.kind:
            case "switch":
                return Switch(value, id=f"set-{s.key}")
            case "mic":
                return Select([(value or "System default", value or "")], value=value or "", allow_blank=False,
                              id="set-mic", compact=True)
            case "key":
                return Button(self.key_text(s.key), id=f"set-{s.key}", compact=True)
            case _ if s.key == "theme":
                return Select([(n, n) for n in self.theme_names()], value=value if value in self.theme_names()
                              else "qal", allow_blank=False, id="set-theme", compact=True)
            case _:
                return Select([(f"{k}  ·  {t.size_gb:g} GB", k) for k, t in TRANSLATION_MODELS.items()],
                              value=value, allow_blank=False, id="set-translation_model", compact=True)

    def theme_names(self) -> list[str]:
        return ["qal", "qal-light", *(n for n in self.app.available_themes if not n.startswith("qal"))]

    def key_text(self, key: str) -> str:
        value = getattr(self.cfg, key)
        return f"{key_label(value) if value else 'Off'} - change"

    def on_mount(self) -> None:
        top = self.query_one(TopBar)
        top.set_status(Content("Settings"))
        top.set_pill("SETTINGS")
        background(lambda: self.post_message(MicsLoaded(input_devices())))

    def on_mics_loaded(self, message: MicsLoaded) -> None:
        current = self.cfg.mic or ""
        options = [("System default", ""), *((n, n) for n in message.names)]
        if current and current not in message.names:
            options.append((f"{current} (from the config)", current))
        select = self.query_one("#set-mic", Select)
        with self.prevent(Select.Changed):
            select.set_options(options)
            select.value = current

    def apply(self, key: str, value, control=None) -> None:
        """Save one setting; on a refusal show why and put the control back."""
        old = getattr(self.cfg, key)
        problem = save_changes(self.app, {key: value})
        self.query_one("#problem", Static).update(Content.styled(problem, "$error") if problem else "")
        if problem:
            if control is not None and not isinstance(control, Button):
                with self.prevent(Switch.Changed, Select.Changed):
                    control.value = old if old is not None else ""
            return
        setting = next(s for s in SETTINGS if s.key == key)
        if key == "theme":
            self.app.theme = value
        elif key == "compact":
            self.app.base.set_class(value, "-compact")
        if not setting.live:
            self.original.setdefault(key, old)
            self.restart_needed = any(getattr(self.cfg, k) != v for k, v in self.original.items())
            self.refresh_bindings()
        if key in ("translation", "translation_model"):
            self.query_one("#help-translation", Static).update(self.help_line(SETTINGS[0]))

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.apply(event.switch.id.removeprefix("set-"), event.value, event.switch)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        key = event.select.id.removeprefix("set-")
        value = event.value or None if key == "mic" else event.value
        if value != getattr(self.cfg, key):
            self.apply(key, value, event.select)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        key = event.button.id.removeprefix("set-")
        translate = key == "translate_key"
        other = self.cfg.dictate_key if translate else self.cfg.translate_key

        def chosen(name: str | None) -> None:
            if name is not None:
                self.apply(key, name, event.button)
                event.button.label = self.key_text(key)

        self.app.push_screen(KeyCapture("Translate" if translate else "Dictate", other, translate), chosen)

    def check_action(self, action: str, parameters: tuple):
        return self.restart_needed if action == "restart" else True

    def action_restart(self) -> None:
        self.app.restart = True
        self.app.action_quit()

    def action_close(self) -> None:
        self.dismiss()


# -- help ---------------------------------------------------------------------------------------------------


class HelpScreen(ModalScreen):
    """Every key that does something, as configured. Any key closes it."""

    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen #help { width: 70; max-width: 100%; max-height: 100%; height: auto; padding: 1 2;
                       background: $surface; border: round $accent; }
    """

    def compose(self) -> ComposeResult:
        app = self.app
        lines = [Content.styled("qal-scribe keys", "bold")]
        if app.hotkeys:
            lines += [Content.assemble(f"Hold {key}", " to ", ("dictate " + app.language_name() if mode == "dictate"
                                                              else "translate to English", "bold"))
                      for key, mode in app.hotkeys.items()]
        else:
            lines += [Content("Run `qal-scribe toggle` to start and stop recording"),
                      Content("`qal-scribe toggle --translate` for English")]
        lines.append(Content(""))
        for b in app.BINDINGS:
            lines.append(Content.assemble((f"{b.key_display or b.key:<8}", "bold $accent"), b.description))
        lines += [Content(""), Content("`qal-scribe doctor` checks everything. Any key closes this.")]
        with Vertical(id="help"):
            yield Static(Content("\n").join(lines))

    def on_key(self, event) -> None:
        event.stop()
        self.dismiss()
