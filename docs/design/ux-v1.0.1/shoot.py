"""Render the TUI states audited in docs/design/ux-v1.0.1.md to SVG (and plain text) with the test FakeController.

Contract: `XDG_CONFIG_HOME=$(mktemp -d) uv run python docs/design/ux-v1.0.1/shoot.py OUT_DIR` from the repo root
writes NAME.svg and NAME.txt per scene; it loads no model and opens no audio device.
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, "tests")
from test_tui import READY, SAMPLE, FakeController, SetupWarnings, done, drive, screen_text, send
from test_tui_screenshots import WARNING, fill_history, speak

from qal_scribe import events as ev
from qal_scribe.config import Config

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ACCESSIBILITY = ("no Accessibility permission: add your terminal app under System Settings > Privacy & Security > "
                 "Accessibility, then quit and reopen it")
INPUT_MONITORING = ("hotkeys need Input Monitoring: add your terminal app under System Settings > Privacy & "
                    "Security > Input Monitoring, then quit and reopen it")


class Downloading(FakeController):
    def start(self):
        self.started.set()
        self.emit(ev.Loading("downloading model 40%"))


class MacPermissions(FakeController):
    def start(self):
        self.started.set()
        self.emit(ev.Notice(ACCESSIBILITY, "warning"))
        self.emit(ev.Notice(INPUT_MONITORING, "warning"))
        self.emit(READY)


class StartFails(FakeController):
    def start(self):
        self.started.set()
        raise ValueError("no input device matches mic = 'Blue Yeti'")


class F13(FakeController):
    def start(self):
        self.started.set()
        self.emit(dataclasses.replace(READY, hotkeys={"f13": "dictate"}))


def shot(name, scene, size=(120, 36), cfg=None, factory=None):
    async def scenario(app, fake, pilot):
        await scene(pilot)
        await pilot.pause(0.2)
        app.save_screenshot(f"{name}.svg", str(OUT))
        (OUT / f"{name}.txt").write_text(screen_text(app))

    drive(scenario, size=size, cfg=cfg, factory=(lambda c, e: factory(c, e)) if factory else None,
          notifications=True)


async def nothing(pilot):
    await pilot.pause()


SCENES = {
    "ready": nothing,
    "recording": speak,
    "recording-translate": lambda pilot: speak(pilot, mode="translate", partial=""),
    "transcribing": lambda pilot: send(pilot, ev.RecordingStarted("dictate"), ev.RecordingStopped("dictate", 4.0)),
    "translating": lambda pilot: send(pilot, ev.RecordingStopped("translate", 4.0),
                                      ev.Loading("loading translation model")),
    "result": lambda pilot: send(pilot, done()),
    "history": fill_history,
    "notice": lambda pilot: send(pilot, ev.Notice(WARNING, "error")),
}
STARTS = {"loading": Downloading, "failed": StartFails, "macperms": MacPermissions, "wayland": SetupWarnings}

for size, tag in [((120, 36), "120"), ((80, 24), "80")]:
    for name, scene in SCENES.items():
        shot(f"{name}-{tag}", scene, size)
    for name, factory in STARTS.items():
        shot(f"{name}-{tag}", nothing, size, factory=factory)
shot("empty-result-120", lambda pilot: send(pilot, done(text="", pasted=False)))
shot("long-result-120", lambda pilot: send(pilot, done(text=" ".join([SAMPLE] * 12))))
shot("f13-120", nothing, factory=F13)
shot("light-history-120", fill_history, cfg=Config(theme="qal-light"))
shot("wide-200x50", fill_history, (200, 50))
shot("short-80x14", fill_history, (80, 14))
for size, tag in [((80, 5), "80x5"), ((40, 3), "40x3")]:
    compact = Config(compact=True)
    shot(f"compact-ready-{tag}", nothing, size, cfg=compact)
    shot(f"compact-recording-{tag}", speak, size, cfg=compact)
    shot(f"compact-loading-{tag}", nothing, size, cfg=compact, factory=Downloading)
    shot(f"compact-perms-{tag}", nothing, size, cfg=compact, factory=MacPermissions)
