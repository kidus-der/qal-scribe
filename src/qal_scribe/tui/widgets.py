"""Widgets for the main window. They only draw what app.py tells them to; app.py owns all state."""
from collections import deque
from datetime import datetime

from rich.segment import Segment
from rich.style import Style
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import ListItem, Static

from qal_scribe import events as ev

STATES = ("loading", "ready", "recording", "transcribing", "translating", "failed", "quitting")
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
BLOCKS = " ▁▂▃▄▅▆▇█"
MODE_NAMES = {"dictate": "Dictate", "translate": "Translate → English"}
HISTORY_LIMIT = 100


def clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def progress_bar(percent: int, width: int = 32) -> tuple[str, str]:
    """(finished part, remaining part) so the caller can colour them differently."""
    done = round(width * min(percent, 100) / 100)
    return "━" * done, "─" * (width - done)


def stats(audio_seconds: float, compute_seconds: float) -> str:
    return f"{audio_seconds:.1f}s · {compute_seconds:.2f}s"


class Meter(Widget):
    """Scrolling mic-level bars, newest on the right. Absolute scale: quiet room is a flat line."""

    COMPONENT_CLASSES = {"meter--low", "meter--high", "meter--clip"}  # noqa: RUF012 (Textual convention)
    DEFAULT_CSS = """
    Meter { height: 3; }
    Meter > .meter--low { color: $primary 40%; }
    Meter > .meter--high { color: $accent; }
    Meter > .meter--clip { color: $error; }
    """
    HISTORY = 400  # wider than any terminal

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.levels: deque[float] = deque([0.0] * self.HISTORY, maxlen=self.HISTORY)

    def push(self, level: float) -> None:
        self.levels.append(min(1.0, max(0.0, level)) ** 0.6)  # lifts quiet speech
        self.refresh()

    def reset(self) -> None:
        self.levels.extend([0.0] * self.HISTORY)
        self.refresh()

    def render_line(self, y: int) -> Strip:
        width, height = self.size
        _, base = self.background_colors
        low, high, clip = (
            (base + self.get_component_styles(f"meter--{name}").color).rich_color for name in ("low", "high", "clip")
        )
        background = base.rich_color
        row_from_bottom = height - 1 - y
        segments = []
        for level in list(self.levels)[-width:]:
            eighths = max(1, round(level * height * 8)) - row_from_bottom * 8  # the baseline is never empty
            color = clip if level > 0.95 else high if level > 0.45 else low
            segments.append(Segment(BLOCKS[min(8, max(0, eighths))], Style(color=color, bgcolor=background)))
        pad = width - len(segments)
        blank = [Segment(" " * pad, Style(bgcolor=background))] if pad > 0 else []
        return Strip([*blank, *segments], width)


class TopBar(Horizontal):
    """Brand, model/device/mic status and the state pill."""

    def compose(self) -> ComposeResult:
        yield Static(Content.assemble(("ቃል", "bold $accent"), (" qal-scribe", "bold")), id="brand")
        yield Static("", id="status")
        yield Static("", id="mic", markup=False)
        yield Static("", id="pill")

    def set_status(self, text: Content, mic: str = "") -> None:
        self.query_one("#status", Static).update(text)
        self.query_one("#mic", Static).update(mic)

    def set_pill(self, label: str) -> None:
        self.query_one("#pill", Static).update(label)


class Reading(VerticalScroll, can_focus=False):
    """Scrolls a long transcript by mouse; keeps keyboard focus on the history list."""


class Stage(Vertical):
    """The centre panel: meter, then the live partial / latest transcript or an empty/loading state."""

    def compose(self) -> ComposeResult:
        yield Meter(id="meter")
        with Reading(id="reading"):
            yield Static("ቃል", id="hero-title")
            yield Static("", id="notes")
            yield Static("", id="hero", markup=False)
            yield Static("", id="transcript", markup=False)

    def show_hero(self, lines: Content, big: bool = True, notes: Content | None = None) -> None:
        self.query_one("#notes", Static).update(notes or "")
        self.query_one("#notes").display = notes is not None
        self.query_one("#hero", Static).update(lines)
        self.query_one("#hero-title").display = big
        self.query_one("#hero").display = True
        self.query_one("#transcript").display = False

    def show_transcript(self, text: str, live: bool = False, empty: bool = False) -> None:
        self.set_class(live, "-live")
        self.set_class(empty, "-empty")
        self.query_one("#transcript", Static).update(text)
        self.query_one("#hero-title").display = False
        self.query_one("#hero").display = False
        self.query_one("#notes").display = False
        self.query_one("#transcript").display = True
        if live:  # a long draft grows downwards: keep its newest words in view (force: compact hides overflow)
            self.query_one("#reading").scroll_end(animate=False, immediate=True, force=True)


class HistoryRow(ListItem):
    """One finished dictation. Keeps the event so Enter can copy it."""

    def __init__(self, entry: ev.Transcribed) -> None:
        super().__init__()
        self.entry = entry
        self.stamp = datetime.now().strftime("%H:%M:%S")

    def compose(self) -> ComposeResult:
        e = self.entry
        yield Static(self.stamp, classes="time")
        yield Static("dictate" if e.mode == "dictate" else "→ EN", classes="mode")
        yield Static(e.text or "(nothing recognised)", classes="text" if e.text else "text empty",
                     markup=False)
        yield Static(stats(e.audio_seconds, e.compute_seconds), classes="stats")
        mark = ("✓", "ok") if e.pasted else ("✗", "no") if e.text else ("·", "none")  # nothing said is not a failure
        yield Static(mark[0], classes=f"paste {mark[1]}")
