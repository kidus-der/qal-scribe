"""Textual TUI (workstream `tui` replaces this placeholder with the designed interface).

Contract: run(cfg) blocks until the user quits. It builds Controller(cfg, emit) where emit hands each Event to
the UI thread via app.call_from_thread, and starts the controller in a worker so the UI paints immediately.
"""
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Log

from qal_scribe import events as ev
from qal_scribe.config import Config
from qal_scribe.controller import Controller
from qal_scribe.plain import format_event


class QalApp(App):
    TITLE = "qal-scribe"
    BINDINGS = [("q", "quit", "Quit")]  # noqa: RUF012 (Textual convention)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        yield Header()
        yield Log()
        yield Footer()

    def on_mount(self) -> None:
        self.controller = Controller(self.cfg, lambda e: self.call_from_thread(self.show, e))
        self.run_worker(self.controller.start, thread=True)

    def show(self, e: ev.Event) -> None:
        line = format_event(e)
        if line:
            self.query_one(Log).write_line(line)


def run(cfg: Config) -> None:
    QalApp(cfg).run()
