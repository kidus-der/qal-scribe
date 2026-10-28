"""`qal-scribe doctor`: one line per check, exit code 1 if anything blocks dictation (workstream `packaging`)."""
from qal_scribe.config import Config


def run(cfg: Config) -> int:
    from qal_scribe import paste

    ok, reason = paste.can_paste()
    print(f"{'ok ' if ok else 'FAIL'}  paste: {reason}")
    return 0 if ok else 1
