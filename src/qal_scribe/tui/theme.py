"""The `qal` themes: one deep-green accent on a near-black (or paper-white) surface, nothing else loud."""
from textual.theme import Theme

QAL_DARK = Theme(
    name="qal",
    primary="#2f9e6e",
    secondary="#5b8fb0",
    accent="#5fd3a0",
    warning="#d9a441",
    error="#e0645c",
    success="#5fd3a0",
    foreground="#d9e3dd",
    background="#0b1210",
    surface="#101a16",
    panel="#16231d",
    boost="#ffffff10",
    dark=True,
    variables={"border": "#24382f", "border-blurred": "#1b2b24", "footer-key-foreground": "#5fd3a0"},
)

QAL_LIGHT = Theme(
    name="qal-light",
    primary="#1d7a4f",
    secondary="#3d6f8f",
    accent="#17653f",
    warning="#a8741a",
    error="#b8423a",
    success="#17653f",
    foreground="#1c2a23",
    background="#f3f7f4",
    surface="#ffffff",
    panel="#e6eee9",
    boost="#00000008",
    dark=False,
    variables={"border": "#b9cdc2", "border-blurred": "#d0ddd5", "footer-key-foreground": "#17653f"},
)

THEMES = (QAL_DARK, QAL_LIGHT)
