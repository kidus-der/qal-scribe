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
PREVIEW_INTERVAL = 0.6  # pause between live previews while recording
STOP_WAIT = 5.0  # seconds stop() waits for a running model call before giving up on unloading
PREVIEW_MAX_SECONDS = 1.5  # slower previews (CPU) would hold up the final result: stop previewing then


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
        self.recorder = self.hotkeys = None
        self._clips: queue.Queue = queue.Queue()
        self._last_level = 0.0
        # _gpu: one model call at a time (final transcription, translation, language swap). A preview only runs
        # when it is free, so it can delay a final result by at most the one preview already in flight.
        self._gpu = threading.Lock()
        self._switching = threading.Lock()
        self._preview_lock = threading.Lock()  # makes "stop previewing" and "emit a Partial" mutually exclusive
        self._preview_stop = threading.Event()
        self._stopped = threading.Event()

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
        self.recorder = Recorder(on_level=self._level, device=self.cfg.mic)
        self.hotkeys = PushToTalk(self.keys(), self._start_recording, self._stop_recording)
        self.hotkeys.start()
        if self.hotkeys.reason:
            self.emit(ev.Notice(self.hotkeys.reason, "warning"))
        ipc.serve(self.toggle)
        threading.Thread(target=self._worker, daemon=True).start()
        self.emit(self._ready())

    def _ready(self) -> ev.Ready:
        return ev.Ready(self.engine.device, self.spec.repo, self.spec.language, self.engine.lm is not None,
                        {key_label(k): m for k, m in self.keys().items()})

    def stop(self) -> None:
        """Shut down: stop previews, hotkeys, IPC and recorder, drop queued clips, free the model and GPU memory.
        Safe to call from the UI thread: it waits only for a model call already running (STOP_WAIT at most), never
        for a UI event."""
        self._stopped.set()
        self._preview_stop.set()
        if self.hotkeys:
            self.hotkeys.stop()
        stop_ipc = getattr(ipc, "stop", None)  # platform workstream; optional until it lands
        if stop_ipc:
            stop_ipc()
        if self.recorder and self.recorder.active:
            self.recorder.stop()
        self._clips.put(None)  # wakes the worker so it exits
        self.translator = None
        if self._gpu.acquire(timeout=STOP_WAIT):
            try:
                self.engine.unload()
            finally:
                self._gpu.release()

    def set_language(self, language: str) -> None:
        """Load `language`'s model in the background (Loading..., then Ready) and swap it in. The current model
        keeps serving until the swap and is freed afterwards. ValueError for an unknown language."""
        spec = models.get(language)
        threading.Thread(target=self._switch, args=(spec,), daemon=True).start()

    def _switch(self, spec: models.ModelSpec) -> None:
        if not self._switching.acquire(blocking=False):
            self.emit(ev.Notice("already switching language", "warning"))
            return
        try:
            if spec != self.spec:
                self.emit(ev.Loading(f"loading {spec.language_name}"))
                new = Engine(spec, device=self.cfg.device, use_lm=self.cfg.lm)
                new.load(lambda m: self.emit(ev.Loading(m)))
                with self._gpu:
                    old, self.engine, self.spec, self.cfg.language = self.engine, new, spec, spec.language
                    old.unload()
            self.emit(self._ready())
        except Exception as e:  # noqa: BLE001 - the old model stays active; report instead of dying in a thread
            self.emit(ev.Notice(f"could not switch to {spec.language_name}: {type(e).__name__}: {e}", "error"))
        finally:
            self._switching.release()

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
        if self.cfg.preview:
            self._preview_stop = stop = threading.Event()
            threading.Thread(target=self._preview_loop, args=(stop,), daemon=True).start()

    def _stop_preview(self) -> None:
        with self._preview_lock:
            self._preview_stop.set()

    def _preview_loop(self, stop: threading.Event) -> None:
        """While recording, show a draft of the audio so far. Gives up for this recording on error or when
        previews are too slow to be worth it."""
        last = ""
        while not stop.wait(PREVIEW_INTERVAL):
            audio = self.recorder.snapshot()
            if len(audio) < SAMPLE_RATE // 2 or np.abs(audio).max() < 1e-3:
                continue
            if not self._gpu.acquire(blocking=False):
                continue
            try:
                if stop.is_set():
                    return
                result = self.engine.preview(audio)
            except Exception as e:  # noqa: BLE001 - previews are optional; never disturb the recording
                self.emit(ev.Notice(f"live preview stopped: {type(e).__name__}: {e}", "warning"))
                return
            finally:
                self._gpu.release()
            with self._preview_lock:
                if stop.is_set():
                    return
                if result.text != last:
                    last = result.text
                    self.emit(ev.Partial(result.text))
            if result.compute_seconds > PREVIEW_MAX_SECONDS:
                return

    def _stop_recording(self, mode: ev.Mode) -> None:
        self._stop_preview()
        audio = self.recorder.stop()
        if self.cfg.sounds:
            play_cue("stop")
        self.emit(ev.RecordingStopped(mode, len(audio) / SAMPLE_RATE))
        self._clips.put((mode, audio))

    def toggle(self, mode: ev.Mode = "dictate") -> None:
        """External trigger (Linux/Wayland `qal-scribe toggle`): start if idle, stop if recording."""
        (self._stop_recording if self.recorder.active else self._start_recording)(mode)

    def _worker(self) -> None:
        while (clip := self._clips.get()) is not None and not self._stopped.is_set():
            mode, audio = clip
            try:
                self.handle(mode, audio)
            except Exception as e:  # noqa: BLE001 - keep the tool alive; show the error instead of dying in a thread
                self.emit(ev.Notice(f"{type(e).__name__}: {e}", "error"))

    def handle(self, mode: ev.Mode, audio: np.ndarray) -> None:
        if len(audio) < SAMPLE_RATE // 4 or np.abs(audio).max() < 1e-3:
            self.emit(ev.Notice("too short or silent; check the terminal's Microphone permission if this repeats",
                                "warning"))
            return
        with self._gpu:
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
