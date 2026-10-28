"""Wires hotkey -> recorder -> engine -> post-processing -> (translation) -> paste, and emits Events.

The UI (app.py or plain.py) constructs Controller(cfg, emit) and calls start(); emit is called from
background threads, so UIs must hand events to their own thread (Textual: app.call_from_thread).
"""
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from qal_scribe import config as config_mod
from qal_scribe import events as ev
from qal_scribe import ipc, models, paste, postprocess
from qal_scribe.audio import SAMPLE_RATE, Recorder, play_cue
from qal_scribe.engine import Engine
from qal_scribe.hotkey import PushToTalk, key_label

LEVEL_INTERVAL = 1 / 25  # throttle mic-level events to ~25/s


def load_words(cfg: config_mod.Config) -> dict[str, str]:
    """words.txt lines: `heard => wanted` replacements; blank lines and # comments ignored."""
    if not cfg.words_file.exists():
        return {}
    pairs = {}
    for line in cfg.words_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=>" in line:
            heard, wanted = (s.strip() for s in line.split("=>", 1))
            if heard:
                pairs[heard] = wanted
    return pairs


class Controller:
    def __init__(self, cfg: config_mod.Config, emit: Callable[[ev.Event], None]):
        self.cfg, self.emit = cfg, emit
        self.spec = models.get(cfg.language)
        self.engine = Engine(self.spec, device=cfg.device, use_lm=cfg.lm)
        self.words = load_words(cfg)
        self.translator = None
        self._clips: queue.Queue = queue.Queue()
        self._last_level = 0.0

    def keys(self) -> dict[str, ev.Mode]:
        keys: dict[str, ev.Mode] = {self.cfg.dictate_key: "dictate"}
        if self.cfg.translate_key:
            keys[self.cfg.translate_key] = "translate"
        return keys

    def start(self) -> None:
        self.engine.load(lambda m: self.emit(ev.Loading(m)))
        ok, reason = paste.can_paste()
        if not ok:
            self.emit(ev.Notice(reason, "warning"))
        self.recorder = Recorder(on_level=self._level)
        self.hotkeys = PushToTalk(self.keys(), self._start_recording, self._stop_recording)
        self.hotkeys.start()
        if self.hotkeys.reason:
            self.emit(ev.Notice(self.hotkeys.reason, "warning"))
        ipc.serve(self.toggle)
        threading.Thread(target=self._worker, daemon=True).start()
        self.emit(ev.Ready(self.engine.device, self.spec.repo, self.spec.language, self.engine.lm is not None,
                           {key_label(k): m for k, m in self.keys().items()}))

    def _level(self, rms: float) -> None:
        now = time.monotonic()
        if now - self._last_level >= LEVEL_INTERVAL:
            self._last_level = now
            self.emit(ev.Level(min(1.0, rms * 4)))

    def _start_recording(self, mode: ev.Mode) -> None:
        self.recorder.start()
        if self.cfg.sounds:
            play_cue("start")
        self.emit(ev.RecordingStarted(mode))

    def _stop_recording(self, mode: ev.Mode) -> None:
        audio = self.recorder.stop()
        if self.cfg.sounds:
            play_cue("stop")
        self.emit(ev.RecordingStopped(mode, len(audio) / SAMPLE_RATE))
        self._clips.put((mode, audio))

    def toggle(self, mode: ev.Mode = "dictate") -> None:
        """External trigger (Linux/Wayland `qal-scribe toggle`): start if idle, stop if recording."""
        (self._stop_recording if self.recorder.active else self._start_recording)(mode)

    def _worker(self) -> None:
        while True:
            mode, audio = self._clips.get()
            try:
                self.handle(mode, audio)
            except Exception as e:  # noqa: BLE001 - keep the tool alive; show the error instead of dying in a thread
                self.emit(ev.Notice(f"{type(e).__name__}: {e}", "error"))

    def handle(self, mode: ev.Mode, audio: np.ndarray) -> None:
        if len(audio) < SAMPLE_RATE // 4 or np.abs(audio).max() < 1e-3:
            self.emit(ev.Notice("too short or silent; check the terminal's Microphone permission if this repeats",
                                "warning"))
            return
        result = self.engine.transcribe(audio)
        text = postprocess.process(result.text, self.cfg, self.words)
        if mode == "translate" and text:
            text = self.translate(text)
        pasted = bool(text) and paste.paste(text + " ", self.cfg.restore_delay)
        self.emit(ev.Transcribed(mode, text, result.text, len(audio) / SAMPLE_RATE, result.compute_seconds, pasted))

    def translate(self, text: str) -> str:
        if self.translator is None:
            from qal_scribe.translate import Translator

            self.emit(ev.Loading("loading translation model"))
            self.translator = Translator(self.engine.device)
            self.translator.load()
        return self.translator.translate(text)
