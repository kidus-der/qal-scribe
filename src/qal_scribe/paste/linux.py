"""Linux (beta): workstream `platform` implements wtype / xdotool / ydotool typing here."""


def can_paste() -> tuple[bool, str]:
    return False, "Linux text insertion is not implemented yet"


def paste(text: str, restore_delay: float = 0.5) -> bool:
    return False
