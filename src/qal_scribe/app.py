"""Textual TUI: header with state pill, live mic meter and transcript stage, history, key-hint footer.

Contract: run(cfg) blocks until the user quits. The Controller is built and started in a thread worker so the
window paints immediately; its events reach the UI thread through app.post_message (thread-safe and
non-blocking, unlike call_from_thread, which would stall the audio callback that emits Level events).
Importing this module must stay torch-free: heavy modules are imported inside the worker.
"""
import os
import re
import sys
import threading
import time
from collections.abc import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import Footer, ListView, Static
from textual.worker import Worker, WorkerState

from qal_scribe import events as ev
from qal_scribe import models
from qal_scribe.config import Config
from qal_scribe.tui import theme
from qal_scribe.tui.widgets import (
    HISTORY_LIMIT,
    MODE_NAMES,
    SPINNER,
    STATES,
    HistoryRow,
    Meter,
    Stage,
    TopBar,
    clock,
    progress_bar,
)

SEVERITY = {"info": "information", "warning": "warning", "error": "error"}
PERCENT = re.compile(r"(\d{1,3})\s*%")
HISTORY_LEGEND = "audio · compute   ✓ pasted  ✗ not pasted"


class ControllerEvent(Message):
    """An Event from the controller, delivered on the UI thread."""

    def __init__(self, event: ev.Event) -> None:
        super().__init__()
        self.event = event


class StartFailed(Message):
    def __init__(self, problem: str) -> None:
        super().__init__()
        self.problem = problem


def mic_name(device: str | int | None = None) -> str | None:
    """Name of the input device the recorder will use (cfg.mic: name substring, index or None = default)."""
    try:
        import sounddevice

        if isinstance(device, str) and device.isdigit():
            device = int(device)
        return sounddevice.query_devices(device, kind="input")["name"]
    except Exception:  # noqa: BLE001 - no audio stack or input device just means no mic name in the header
        return None


def make_controller(cfg: Config, emit: Callable[[ev.Event], None]):
    from qal_scribe.controller import Controller

    return Controller(cfg, emit)


class QalApp(App):
    TITLE = "qal-scribe"
    CSS_PATH = "tui/qal.tcss"
    ENABLE_COMMAND_PALETTE = False
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (110, "-wide")]  # noqa: RUF012 (Textual convention)
    VERTICAL_BREAKPOINTS = [(0, "-tiny"), (4, "-small")]  # noqa: RUF012 (-tiny: compact drops the footer)
    BINDINGS = [  # noqa: RUF012 (Textual convention)
        Binding("c", "copy", "Copy last"),
        Binding("l", "language", "Language"),
        Binding("t", "cycle_theme", "Theme"),
        Binding("m", "compact", "Compact"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, cfg: Config, controller_factory: Callable = make_controller):
        super().__init__()
        self.cfg = cfg
        self.controller_factory = controller_factory
        self.controller = None
        self.boot_thread: threading.Thread | None = None
        self.info: ev.Ready | None = None
        self.language = cfg.language
        self.mic: str | None = None
        self.failure = ""  # set when the controller could not start; shown instead of loading forever
        self.loading = True  # until Ready; also true again while a language switch reloads the model
        self.loading_message = "starting"
        self.rec_mode: ev.Mode | None = None
        self.rec_started = 0.0
        self.pending: list[ev.Mode] = []  # clips recorded but not yet transcribed, oldest first
        self.partial = ""
        self.last: ev.Transcribed | None = None
        self.spin = 0
        self.quitting = False

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        yield Stage(id="stage")
        with Vertical(id="history-panel"):
            yield ListView(id="history")
            yield Static("Your dictations will collect here, newest first.", id="history-empty")
        with Horizontal(id="bottom"):
            yield Footer(show_command_palette=False)
            yield Static("", id="keys")

    def on_mount(self) -> None:
        for t in theme.THEMES:
            self.register_theme(t)
        if self.cfg.theme in self.available_themes:
            self.theme = self.cfg.theme
        else:
            self.theme = "qal"
            self.notify(f"unknown theme {self.cfg.theme!r}; using qal", severity="warning")
        self.screen.set_class(self.cfg.compact, "-compact")
        self.query_one("#history-panel").border_title = "History"
        self.set_interval(0.1, self.tick)
        mic = getattr(self.cfg, "mic", None)  # Config.mic arrives with the platform workstream
        self.run_worker(lambda: mic_name(mic), thread=True, name="mic", group="mic", exit_on_error=False)
        # A daemon thread, not a Textual worker: those are joined at exit, which would hold `q` hostage
        # until a first-run model download finishes.
        self.boot_thread = threading.Thread(target=self.boot, name="controller-start", daemon=True)
        self.boot_thread.start()
        self.refresh_view()

    def boot(self) -> None:
        def emit(event: ev.Event) -> None:
            self.post_message(ControllerEvent(event))

        try:
            self.controller = self.controller_factory(self.cfg, emit)
            self.controller.start()
        except Exception as e:  # noqa: BLE001 - whatever stopped the model or hotkeys starting goes on screen
            self.post_message(StartFailed(f"{type(e).__name__}: {e}"))

    # -- events from the controller -------------------------------------------------------------------------

    async def on_controller_event(self, message: ControllerEvent) -> None:
        match message.event:
            case ev.Level(rms):
                if self.rec_mode:
                    self.query_one(Meter).push(rms)
                return
            case ev.Loading(text):
                self.loading_message = text
                self.loading = self.loading or not (self.rec_mode or self.pending)  # else: translation model
            case ev.Ready() as ready:
                self.info, self.language = ready, ready.language
                self.loading, self.loading_message = False, ""
            case ev.RecordingStarted(mode):
                self.rec_mode, self.rec_started, self.partial = mode, time.monotonic(), ""
                self.query_one(Meter).reset()
            case ev.Partial(text):
                self.partial = text
            case ev.RecordingStopped(mode):
                self.rec_mode = None
                self.pending.append(mode)
                self.query_one(Meter).reset()
            case ev.Transcribed() as done:
                await self.finish(done)
            case ev.Notice(text, level):
                if level != "info" and self.pending:  # a skipped or failed clip never produces Transcribed
                    self.pending.pop(0)
                self.notify(text, severity=SEVERITY[level], timeout=10 if level == "error" else 5)
        self.refresh_view()

    async def finish(self, done: ev.Transcribed) -> None:
        if self.pending:
            self.pending.pop(0)
        self.last, self.loading_message = done, ""
        history = self.query_one(ListView)
        await history.insert(0, [HistoryRow(done)])
        if len(history) > HISTORY_LIMIT:
            await history.pop()
        if self.focused is None:  # the list is hidden (so unfocusable) until the first entry exists
            history.focus()

    def on_start_failed(self, message: StartFailed) -> None:
        self.failure = message.problem
        self.notify(message.problem, severity="error", timeout=10)
        self.refresh_view()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        worker = event.worker
        if worker.name == "mic" and worker.state is WorkerState.SUCCESS:
            self.mic = worker.result
        elif worker.name == "stop" and worker.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            self.exit()
        elif worker.state is WorkerState.ERROR:
            self.notify(f"{type(worker.error).__name__}: {worker.error}", severity="error", timeout=10)
        self.refresh_view()

    # -- state -> pixels ------------------------------------------------------------------------------------

    @property
    def hotkeys(self) -> dict[str, ev.Mode]:
        return self.info.hotkeys if self.info else {}

    @property
    def state(self) -> str:
        if self.failure:
            return "failed"
        if self.rec_mode:
            return "recording"
        if self.pending:
            return "translating" if self.pending[0] == "translate" else "transcribing"
        return "loading" if self.loading else "ready"

    def tick(self) -> None:
        self.spin = (self.spin + 1) % len(SPINNER)
        if self.state in ("loading", "recording", "transcribing", "translating"):
            self.refresh_view()

    def pill_label(self) -> str:
        spinner = SPINNER[self.spin]
        match self.state:
            case "loading":
                return f"{spinner} LOADING"
            case "recording":
                to_english = " → EN" if self.rec_mode == "translate" else ""
                return f"● REC{to_english} {clock(time.monotonic() - self.rec_started)}"
            case "transcribing":
                return f"{spinner} TRANSCRIBING"
            case "translating":
                return f"{spinner} TRANSLATE"
            case "failed":
                return "ERROR"
        return "READY"

    def status_line(self) -> Content:
        if not self.info:
            return Content("")
        name = models.MODELS[self.info.language].language_name if self.info.language in models.MODELS else ""
        model = self.info.model.rsplit("/", 1)[-1]
        parts = [name, model, self.info.device, "LM on" if self.info.lm else "LM off"]
        if self.mic:
            parts.append(f"mic {self.mic}")
        return Content(" · ".join(p for p in parts if p))

    def hero(self) -> Content:
        if self.failure:
            return Content.assemble(("Could not start\n", "bold $error"), self.failure, "\nPress q to quit.")
        if self.loading or not self.info:
            parts: list = [f"{SPINNER[self.spin]}  {self.loading_message}"]
            if percent := PERCENT.search(self.loading_message):
                done, rest = progress_bar(int(percent[1]))
                parts += ["\n", (done, "$accent"), (rest, "$border")]
            return Content.assemble(*parts)
        if self.pending:
            verb = "Translating" if self.pending[0] == "translate" else "Transcribing"
            return Content(f"{SPINNER[self.spin]}  {verb}…")
        verbs = {"dictate": "dictate Amharic", "translate": "translate to English"}
        lines = [Content.assemble("Hold ", (key, "bold $accent"), f" to {verbs[mode]}")
                 for key, mode in self.hotkeys.items()]
        return Content("\n").join([*lines, Content("Release to paste at your cursor.")])

    def refresh_view(self) -> None:
        state = self.state
        for s in STATES:
            self.screen.set_class(s == state, f"-{s}")
        top = self.query_one(TopBar)
        top.set_pill(self.pill_label())
        top.set_status(self.status_line())
        keys = "  ".join(f"{key} {mode}" for key, mode in self.hotkeys.items())
        self.query_one("#keys", Static).update(keys)
        stage = self.query_one(Stage)
        stage.border_title = "Starting" if state in ("loading", "failed") else MODE_NAMES[self.active_mode()]
        stage.border_subtitle = self.stage_subtitle()
        if self.rec_mode:
            stage.show_transcript(self.partial or "listening…", live=True)
        elif self.last and state in ("ready", "transcribing", "translating"):
            stage.show_transcript(self.last.text or "(nothing recognised)")
        else:
            stage.show_hero(self.hero(), big=not self.screen.has_class("-compact"))
        history = self.query_one(ListView)
        history.display = bool(len(history))
        self.query_one("#history-empty").display = not len(history)
        self.query_one("#history-panel").border_subtitle = HISTORY_LEGEND if len(history) else ""

    def active_mode(self) -> ev.Mode:
        if self.rec_mode or self.pending:
            return self.rec_mode or self.pending[0]
        return self.last.mode if self.last else "dictate"

    def stage_subtitle(self) -> str:
        if self.rec_mode:
            key = next((k for k, m in self.hotkeys.items() if m == self.rec_mode), "key")
            return f"release {key} to finish"
        if self.pending:
            return self.loading_message
        if self.last and self.state == "ready":
            e = self.last
            pasted = "pasted" if e.pasted else "not pasted"
            return f"{e.audio_seconds:.1f}s audio · {e.compute_seconds:.2f}s compute · {pasted}"
        return ""

    # -- actions --------------------------------------------------------------------------------------------

    def action_copy(self) -> None:
        if self.last and self.last.text:
            self.copy(self.last.text)
        else:
            self.notify("nothing to copy yet", severity="warning", timeout=2)

    def copy(self, text: str) -> None:
        self.copy_to_clipboard(text)
        self.notify("copied", timeout=1.5)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, HistoryRow) and event.item.entry.text:
            self.copy(event.item.entry.text)

    def action_language(self) -> None:
        codes = list(models.MODELS)
        switch = getattr(self.controller, "set_language", None)
        if len(codes) < 2:
            self.notify(f"only {models.MODELS[codes[0]].language_name} is installed", timeout=3)
        elif switch is None:
            self.notify("language switching is not available yet", severity="warning", timeout=3)
        else:
            here = codes.index(self.language) if self.language in codes else -1
            code = codes[(here + 1) % len(codes)]
            self.notify(f"language: {models.MODELS[code].language_name}", timeout=3)
            self.run_worker(lambda: switch(code), thread=True, name="language", group="language",
                            exclusive=True, exit_on_error=False)

    def action_cycle_theme(self) -> None:
        names = ["qal", "qal-light", *(n for n in self.available_themes if not n.startswith("qal"))]
        self.theme = names[(names.index(self.theme) + 1) % len(names)] if self.theme in names else "qal"
        self.notify(self.theme, title="theme", timeout=1.5)

    def action_compact(self) -> None:
        self.screen.toggle_class("-compact")
        self.refresh_view()

    def action_quit(self) -> None:
        if self.quitting:
            return
        self.quitting = True
        stop = getattr(self.controller, "stop", None)
        self.run_worker(stop or (lambda: None), thread=True, name="stop", group="stop", exit_on_error=False)


def run(cfg: Config) -> None:
    app = QalApp(cfg)
    app.run()
    if app.boot_thread and app.boot_thread.is_alive():
        # shortcut: quitting mid-download; skip interpreter teardown so torch/requests threads cannot abort it
        sys.stdout.flush()
        os._exit(0)
