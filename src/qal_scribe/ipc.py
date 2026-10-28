"""Local control socket so `qal-scribe toggle` can drive a running instance (workstream `platform`)."""


def serve(on_toggle) -> None:
    """Start a background listener; each `qal-scribe toggle [--translate]` calls on_toggle(mode)."""


def send_toggle() -> int:
    print("toggle is not implemented yet")
    return 1
