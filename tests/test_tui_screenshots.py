"""SVG screenshots of the main states. Set QAL_TUI_SHOTS to a folder to keep them (default: tmp_path)."""
import math
import os
from pathlib import Path

import pytest
from test_tui import SAMPLE, done, drive, screen_text, send

from qal_scribe import events as ev
from qal_scribe.config import Config

SIZES = {"120x36": (120, 36), "80x24": (80, 24), "compact-80x5": (80, 5), "compact-40x3": (40, 3)}
LONG = "ይህ ረጅም ዓረፍተ ነገር ነው ብዙ ቃላት ያሉት ሁለት መስመር የሚሞላ እና የሚቀጥል ለማየት ያህል ተጽፎ ዛሬ"
HISTORY = ["እንኳን ደህና መጣህ", "አዲስ አበባ ዛሬ ፀሐያማ ናት።", LONG, SAMPLE]


async def speak(pilot, mode="dictate", partial=SAMPLE):
    levels = [ev.Level(abs(math.sin(i / 6)) * 0.3 * (1 + (i % 7) / 6)) for i in range(140)]
    await send(pilot, ev.RecordingStarted(mode), *levels, ev.Partial(partial))
    pilot.app.rec_started -= 4
    pilot.app.tick()
    await pilot.pause()


async def fill_history(pilot):
    entries = [
        done(t, mode="translate" if i % 2 else "dictate", pasted=i != 2, audio=2.5 + i, compute=0.1 + i / 50)
        for i, t in enumerate(HISTORY)
    ]
    await send(pilot, *entries)
    await pilot.press("down", "down")  # leaves the second row highlighted


SCENES = {
    "ready": lambda pilot: pilot.pause(),
    "recording": speak,
    "transcribing": lambda pilot: send(pilot, ev.RecordingStopped("dictate", 4.0)),
    "translating": lambda pilot: send(pilot, ev.RecordingStopped("translate", 4.0)),
    "transcript": fill_history,
}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("scene", SCENES)
def test_screenshot(scene, size, tmp_path):
    folder = Path(os.environ.get("QAL_TUI_SHOTS", tmp_path))
    folder.mkdir(parents=True, exist_ok=True)

    async def scenario(app, fake, pilot):
        await SCENES[scene](pilot)
        await pilot.pause()
        assert screen_text(app).strip()
        app.save_screenshot(f"{scene}-{size}.svg", str(folder))

    drive(scenario, size=SIZES[size], cfg=Config(compact=size.startswith("compact")))
    assert (folder / f"{scene}-{size}.svg").stat().st_size > 5_000


def test_loading_screenshot(tmp_path):
    folder = Path(os.environ.get("QAL_TUI_SHOTS", tmp_path))
    folder.mkdir(parents=True, exist_ok=True)

    async def scenario(app, fake, pilot):
        assert fake.started.wait(5)
        await send(pilot, ev.Loading("downloading model 40%"))
        app.save_screenshot("loading-120x36.svg", str(folder))
        fake.hold.set()

    drive(scenario, hold=True)
    assert (folder / "loading-120x36.svg").stat().st_size > 5_000
