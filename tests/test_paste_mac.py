import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS clipboard")


def test_clipboard_round_trip_on_private_pasteboard():
    from AppKit import NSData, NSPasteboard, NSPasteboardItem, NSPasteboardTypeString

    from qal_scribe.paste.mac import restore_clipboard, save_clipboard

    def data(b):
        return NSData.dataWithBytes_length_(b, len(b))

    pb = NSPasteboard.pasteboardWithUniqueName()  # never the user's real clipboard
    try:
        a = NSPasteboardItem.alloc().init()
        a.setString_forType_("ሰላም plain", NSPasteboardTypeString)
        a.setData_forType_(data("{\\rtf1 ሰላም}".encode()), "public.rtf")
        a.setData_forType_(data(b"\x89PNG fake\x00\x01"), "public.png")
        b = NSPasteboardItem.alloc().init()
        b.setString_forType_("second", NSPasteboardTypeString)
        pb.clearContents()
        pb.writeObjects_([a, b])
        before = save_clipboard(pb)
        pb.clearContents()
        pb.setString_forType_("transcript", NSPasteboardTypeString)
        restore_clipboard(pb, before)
        after = save_clipboard(pb)
        assert [sorted(i) for i in after] == [sorted(i) for i in before]
        assert all(bytes(x[t]) == bytes(y[t]) for x, y in zip(before, after) for t in x)
    finally:
        pb.releaseGlobally()
