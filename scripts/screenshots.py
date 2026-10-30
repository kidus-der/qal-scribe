"""Regenerate the README screenshots: docs/assets/screens/*.svg and the animated docs/assets/demo.svg.

    uv run python scripts/screenshots.py

Contract: drives the real app through a scripted session (setup, ready, recording, live preview, transcript,
translation, settings) with a fake controller, so no model, microphone or hotkey is touched; writes only under
docs/assets, and never reads or writes the user's real config.
"""
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))  # the FakeController and drive() the TUI tests use
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="qal-shots-")

import test_tui

from qal_scribe import doctor, models
from qal_scribe import events as ev
from qal_scribe.tui import screens

OUT = ROOT / "docs" / "assets"
SIZE = (100, 30)
SECONDS = 2  # per frame in demo.svg
FRAMES = ["setup", "ready", "recording", "transcript", "translation", "settings"]
HELLO = "ሰላም፣ ዛሬ ስብሰባው በስንት ሰዓት ነው?"
HELLO_EN = "Hello, what time is the meeting today?"
EARLIER = "እባክህ ሪፖርቱን ዛሬ ከሰዓት በኋላ ላክልኝ።"


async def session(app, pilot, theme: str, shots: Path) -> None:
    """One pass through the scenes; each is saved as <shots>/<scene>-<theme>.svg."""
    shot = lambda scene: app.save_screenshot(f"{scene}-{theme}.svg", str(shots))
    app.theme = theme
    await pilot.pause()
    shot("setup")
    await pilot.press("c")
    await test_tui.settle(pilot)
    await test_tui.send(pilot, test_tui.READY, ev.Notice("ready", "info"))
    app.clear_notifications()
    await pilot.pause()
    shot("ready")
    await test_tui.send(pilot, ev.RecordingStarted("dictate"),
                        *(ev.Level(abs(math.sin(i / 6)) * 0.35 * (1 + (i % 7) / 6)) for i in range(140)),
                        ev.Partial("ሰላም፣ ዛሬ ስብሰባው በስንት"))
    app.rec_started -= 3
    app.tick()
    await pilot.pause()
    shot("recording")
    await test_tui.send(pilot, ev.RecordingStopped("dictate", 4.2),
                        test_tui.done(EARLIER, audio=3.1, compute=0.11), test_tui.done(HELLO, audio=4.2, compute=0.14))
    shot("transcript")
    await test_tui.send(pilot, ev.RecordingStarted("translate"), ev.RecordingStopped("translate", 3.4),
                        ev.Transcribed("translate", HELLO_EN, HELLO, 3.4, 0.9, True))
    shot("translation")
    await pilot.press("s")
    await pilot.pause()
    shot("settings")
    await pilot.press("escape")


def capture(theme: str, shots: Path) -> None:
    async def scenario(app, fake, pilot):
        await session(app, pilot, theme, shots)

    # a clean first run: nothing downloaded, so the setup screen opens
    with (mock.patch.object(doctor, "is_cached", lambda *a: False),
          mock.patch.object(screens, "system_checks", lambda cfg: []),
          mock.patch.object(models, "missing_bytes", lambda *a: 2_900_000_000)):
        test_tui.drive(scenario, size=SIZE, setup=True, notifications=True)


def demo(shots: Path) -> str:
    """Stack the frames as nested <svg>s and fade them in turn: one file GitHub shows in an <img>."""
    n = len(FRAMES)
    css = (f"@keyframes show{{0%{{opacity:1}}{100 / n:.3f}%{{opacity:0}}100%{{opacity:0}}}}"
           f".f{{opacity:0;animation:show {n * SECONDS}s step-end infinite}}")
    groups = []
    for i, scene in enumerate(FRAMES):
        svg = (shots / f"{scene}-qal.svg").read_text()
        svg = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
        groups.append(f'<g class="f" style="animation-delay:{i * SECONDS}s">{svg}</g>')
    view = re.search(r'viewBox="([^"]+)"', (shots / "ready-qal.svg").read_text()).group(1)
    w, h = view.split()[2:]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" width="{w}" height="{h}">'
            f"<style>{css}</style>{''.join(groups)}</svg>")


def main() -> None:
    shots = OUT / "screens"
    shots.mkdir(parents=True, exist_ok=True)
    for theme in ("qal", "qal-light"):
        capture(theme, shots)
    (OUT / "demo.svg").write_text(demo(shots))
    print(f"wrote {len(list(shots.glob('*.svg')))} screenshots and demo.svg ({(OUT / 'demo.svg').stat().st_size // 1000} kB)")


if __name__ == "__main__":
    main()
