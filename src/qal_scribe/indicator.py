"""macOS recording indicator: a small floating pill near the bottom centre of the main screen.

Contract: the app side is `Indicator(on_error)`: call it with every Controller event and close() it at exit; it
starts `python -m qal_scribe.indicator` and writes one state line per change to its stdin ("listening",
"transcribing" or "hidden"). The helper shows the pill for those states and exits when stdin closes. A helper
that cannot start or dies calls on_error(message) once and the Indicator goes quiet: dictation never depends on it.
"""
import subprocess
import sys
import threading
from collections.abc import Callable

from qal_scribe import events as ev

STATES = ("listening", "transcribing", "hidden")


class Indicator:
    def __init__(self, on_error: Callable[[str], None], popen=subprocess.Popen):
        self.on_error = on_error
        self.recording, self.pending, self.state = False, 0, "hidden"  # pending: clips recorded, not yet finished
        self._lock = threading.Lock()
        self._proc = None
        try:
            self._proc = popen([sys.executable, "-m", "qal_scribe.indicator"], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        except OSError as e:
            self._fail(f"{type(e).__name__}: {e}")

    def __call__(self, event: ev.Event) -> None:
        with self._lock:
            match event:
                case ev.RecordingStarted():
                    self.recording = True
                case ev.RecordingStopped():
                    self.recording, self.pending = False, self.pending + 1
                case ev.Transcribed() | ev.Notice(clip=True):  # every recorded clip ends in one of these
                    self.pending = max(0, self.pending - 1)
                case _:
                    return
            state = "listening" if self.recording else "transcribing" if self.pending else "hidden"
            if state != self.state:
                self.state = state
                self._send(state)

    def _send(self, line: str) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):  # BrokenPipeError: the helper died; ValueError: closed
            self._fail("the helper exited")

    def _fail(self, why: str) -> None:
        self._proc = None
        self.on_error(f"recording indicator unavailable ({why}); dictation works without it")

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc:
            try:
                proc.stdin.close()
            except OSError:
                pass  # already dead; nothing to hide


def _main() -> None:
    """The helper: AppKit's run loop owns this main thread, stdin is read on a second thread."""
    import Quartz  # noqa: F401 - registers the CGColor type, or layer colours come back as raw pointers
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSColor,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSMutableAttributedString,
        NSPanel,
        NSScreen,
        NSStatusWindowLevel,
        NSTextField,
        NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorIgnoresCycle,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSMakeRect
    from PyObjCTools import AppHelper

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon, not in Cmd-Tab
    height, pad, gap_from_bottom = 34.0, 18.0, 56.0
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, 160, height), NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
        NSBackingStoreBuffered, False)
    panel.setLevel_(NSStatusWindowLevel)  # above full-screen apps' content; plain floating can sit below it
    panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
                                 | NSWindowCollectionBehaviorStationary | NSWindowCollectionBehaviorIgnoresCycle)
    panel.setIgnoresMouseEvents_(True)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setHasShadow_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 160, height))
    view.setWantsLayer_(True)
    view.layer().setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.08, 0.86).CGColor())
    view.layer().setCornerRadius_(height / 2)
    panel.setContentView_(view)
    font = NSFont.systemFontOfSize_weight_(14, 0.3)
    label = None

    def text(state: str):
        s = NSMutableAttributedString.alloc().initWithString_("● Listening" if state == "listening" else "Transcribing…")
        s.addAttribute_value_range_(NSFontAttributeName, font, (0, s.length()))
        s.addAttribute_value_range_(NSForegroundColorAttributeName, NSColor.whiteColor(), (0, s.length()))
        if state == "listening":
            s.addAttribute_value_range_(NSForegroundColorAttributeName, NSColor.systemRedColor(), (0, 1))
        return s

    def show(state: str) -> None:
        nonlocal label
        if state == "hidden":
            panel.orderOut_(None)
            return
        if label:
            label.removeFromSuperview()
        label = NSTextField.labelWithAttributedString_(text(state))
        size = label.frame().size
        width = size.width + 2 * pad
        label.setFrameOrigin_((pad, (height - size.height) / 2))
        view.setFrameSize_((width, height))
        view.addSubview_(label)
        screen = NSScreen.mainScreen().visibleFrame()
        panel.setFrame_display_(NSMakeRect(screen.origin.x + (screen.size.width - width) / 2,
                                           screen.origin.y + gap_from_bottom, width, height), True)
        panel.orderFrontRegardless()  # never makeKey: the user's app keeps focus

    def read() -> None:
        for line in sys.stdin:
            if (state := line.strip()) in STATES:
                AppHelper.callAfter(show, state)
        AppHelper.callAfter(app.terminate_, None)

    threading.Thread(target=read, daemon=True).start()
    app.run()


if __name__ == "__main__":
    _main()
