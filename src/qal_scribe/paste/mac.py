"""macOS: clipboard + synthesized ⌘V, restoring the user's clipboard afterwards."""
import time

import Quartz
from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString
from ApplicationServices import AXIsProcessTrusted

V_KEYCODE = 9  # physical V position, so ⌘V works whatever keyboard layout is active


def can_paste() -> tuple[bool, str]:
    if AXIsProcessTrusted():
        return True, "Accessibility permission granted"
    return False, ("no Accessibility permission: add your terminal app under System Settings > "
                   "Privacy & Security > Accessibility, then quit and reopen it")


def save_clipboard(pb) -> list[dict]:
    # Every item in every format (text, rich text, images, files), not just plain text.
    return [{t: item.dataForType_(t) for t in item.types()} for item in pb.pasteboardItems() or []]


def restore_clipboard(pb, saved: list[dict]) -> None:
    pb.clearContents()
    items = []
    for formats in saved:
        item = NSPasteboardItem.alloc().init()
        for t, data in formats.items():
            if data is not None:  # lazily-promised data the source app never supplied
                item.setData_forType_(data, t)
        items.append(item)
    if items:
        pb.writeObjects_(items)


def paste(text: str, restore_delay: float = 0.5, pb=None) -> bool:
    if not AXIsProcessTrusted():
        return False
    pb = pb or NSPasteboard.generalPasteboard()
    saved = save_clipboard(pb)
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    pb.setString_forType_("", "org.nspasteboard.TransientType")  # clipboard managers skip transient items
    ours = pb.changeCount()
    for down in (True, False):
        e = Quartz.CGEventCreateKeyboardEvent(None, V_KEYCODE, down)
        Quartz.CGEventSetFlags(e, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
    time.sleep(restore_delay)  # apps read the clipboard a moment after ⌘V; restoring sooner pastes old contents
    if pb.changeCount() == ours:  # don't clobber something the user copied in the meantime
        restore_clipboard(pb, saved)
    return True
