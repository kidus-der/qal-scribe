"""Insert text at the cursor of the focused app.

Contract: paste(text, restore_delay) -> bool (False when insertion is not possible, e.g. missing permission).
can_paste() -> (ok, reason) for `qal-scribe doctor`.
"""
import sys

if sys.platform == "darwin":
    from qal_scribe.paste.mac import can_paste, paste
elif sys.platform.startswith("linux"):
    from qal_scribe.paste.linux import can_paste, paste
else:
    def can_paste() -> tuple[bool, str]:
        return False, f"pasting is not supported on {sys.platform} yet"

    def paste(text: str, restore_delay: float = 0.5) -> bool:
        return False

__all__ = ["can_paste", "paste"]
