"""Wires hotkey -> recorder -> engine -> post-processing -> (translation) -> paste, and emits Events.

The UI (app.py or plain.py) constructs Controller(cfg, emit) and calls start(); emit is called from
background threads, so UIs must hand events to their own thread (Textual: app.call_from_thread).
stop() returns False when a model call was still running after STOP_WAIT: the UI should then end the process
with os._exit, since Python would wait at exit for that call's threads (a model download, for one).
"""
import contextlib
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from qal_scribe import config as config_mod
from qal_scribe import events as ev
from qal_scribe import ipc, models, paste, postprocess
from qal_scribe.audio import SAMPLE_RATE, Recorder, play_cue
from qal_scribe.config import load_words
from qal_scribe.engine import Engine
from qal_scribe.hotkey import PushToTalk, key_label
from qal_scribe.translate import NLLB_CODES

LEVEL_INTERVAL = 1 / 25  # throttle mic-level events to ~25/s
PREVIEW_INTERVAL = 0.6  # pause between live previews while recording
STOP_WAIT = 5.0  # seconds stop() waits for a running model call before giving up on unloading
PREVIEW_MAX_SECONDS = 1.5  # slower previews (CPU) would hold up the final result: stop previewing then


class Controller:
    def __init__(self, cfg: config_mod.Config, emit: Callable[[ev.Event], None]):
        self.cfg, self.emit = cfg, emit
        self.spec = models.get(cfg.language)
        self.engine = Engine(self.spec, device=cfg.device, use_lm=cfg.lm)
        self.words = load_words(cfg)
        self.translator = None
        self.recorder = self.hotkeys = None
        self._mode: ev.Mode = "dictate"  # mode of the recording in progress
        self._clips: queue.Queue = queue.Queue()
        self._last_level = 0.0
        # _gpu: one model call at a time (final transcription, translation, language swap). A preview only runs
        # when it is free, so it can delay a final result by at most the one preview already in flight.
        self._gpu = threading.Lock()
        self._switching = threading.Lock()
        self._preview_lock = threading.Lock()  # makes "stop previewing" and "emit a Partial" mutually exclusive
        self._rec_lock = threading.Lock()  # the hotkey thread and `qal-scribe toggle` (IPC thread) both start/stop
        self._paste_lock = threading.Lock()  # held while a paste is in flight, so stop() can let it restore
        self._preview_stop = threading.Event()
        self._stopped = threading.Event()

    def keys(self) -> dict[str, ev.Mode]:
        keys: dict[str, ev.Mode] = {self.cfg.dictate_key: "dictate"}
        if self.cfg.translate_key:
            keys[self.cfg.translate_key] = "translate"
        return keys

    def start(self) -> None:
        # Fail before the slow model load: an unknown `mic` (ValueError) or another running instance (RuntimeError).
        self.recorder = Recorder(on_level=self._level, device=self.cfg.mic)
        ipc.serve(self.toggle)
        self.engine.load(lambda m: self.emit(ev.Loading(m)))
        ok, reason = paste.can_paste()
        if not ok:
            self.emit(ev.Notice(reason, "warning"))
        self.hotkeys = PushToTalk(self.keys(), self._start_recording, self._stop_recording)
        self.hotkeys.start()
        if self.hotkeys.reason:
            self.emit(ev.Notice(self.hotkeys.reason, "warning"))
        threading.Thread(target=self._worker, daemon=True).start()
        self.emit(self._ready())

    def _ready(self) -> ev.Ready:
        keys = self.keys() if self.hotkeys and self.hotkeys.listening else {}  # Wayland: only `toggle` works
        return ev.Ready(self.engine.device, self.spec.repo, self.spec.language, self.engine.lm is not None,
                        {key_label(k): m for k, m in keys.items()})

    def stop(self) -> bool:
        """Shut down: stop previews, hotkeys, IPC and recorder, drop queued clips, free the model and GPU memory.
        Safe to call from the UI thread: it waits only for a model call already running (STOP_WAIT at most), never
        for a UI event. False when that call was still running (see the module docstring)."""
        self._stopped.set()
        self._preview_stop.set()
        if self.hotkeys:
            self.hotkeys.stop()
        ipc.stop()
        if self.recorder and self.recorder.active:
            self.recorder.stop()
        self._clips.put(None)  # wakes the worker so it exits
        self.translator = None
        if self._paste_lock.acquire(timeout=self.cfg.restore_delay + 1):  # a paste in flight restores the clipboard
            self._paste_lock.release()
        if not self._gpu.acquire(timeout=STOP_WAIT):
            return False
        try:
            self.engine.unload()
        finally:
            self._gpu.release()
        return True

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
            self.emit(self._ready())  # the old model still serves: the UI must leave its loading state
        finally:
            self._switching.release()

    def _level(self, rms: float) -> None:
        now = time.monotonic()
        if now - self._last_level >= LEVEL_INTERVAL:
            self._last_level = now
            self.emit(ev.Level(min(1.0, rms * 4)))

    def _start_recording(self, mode: ev.Mode) -> None:
        """Hotkey press or toggle. Never raises: on pynput's thread an exception silently kills the hotkeys."""
        with self._rec_lock:
            if self.recorder.active:  # e.g. a hotkey press while a `toggle` recording runs
                return
            try:
                self.recorder.start()
            except Exception as e:  # noqa: BLE001 - e.g. PortAudioError after the mic was unplugged
                self.emit(ev.Notice(f"could not start recording: {type(e).__name__}: {e}; `qal-scribe doctor` "
                                    "checks the microphone", "error"))
                return
            self._mode = mode
            if self.cfg.sounds:
                play_cue("start")
            self.emit(ev.RecordingStarted(mode))
            if mode == "translate" and self.translator is None and self.spec.language in NLLB_CODES:
                threading.Thread(target=self._preload_translator, daemon=True).start()
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

    def _stop_recording(self, *_) -> None:
        """Hotkey release or toggle; keeps the mode the recording started with. Never raises (see above)."""
        with self._rec_lock:
            if not self.recorder.active:  # e.g. the key is released after `toggle` already stopped the recording
                return
            self._stop_preview()
            mode = self._mode
            try:
                audio = self.recorder.stop()
            except Exception as e:  # noqa: BLE001 - same as _start_recording
                self.emit(ev.RecordingStopped(mode, 0.0))
                self.emit(ev.Notice(f"could not stop recording: {type(e).__name__}: {e}", "error", clip=True))
                return
            if self.cfg.sounds:
                play_cue("stop")
            self.emit(ev.RecordingStopped(mode, len(audio) / SAMPLE_RATE))
            self._clips.put((mode, audio))

    def toggle(self, mode: ev.Mode = "dictate") -> None:
        """External trigger (Linux/Wayland `qal-scribe toggle`): start if idle, stop if recording. A stop keeps
        the mode the recording started with, so `toggle --translate` then `toggle` still translates."""
        if self.hotkeys is None:
            raise RuntimeError("still loading the model; try again in a moment")  # the client prints this
        if self.recorder.active:
            self._stop_recording()
        else:
            self._start_recording(mode)

    def _worker(self) -> None:
        while (clip := self._clips.get()) is not None and not self._stopped.is_set():
            mode, audio = clip
            try:
                self.handle(mode, audio)
            except Exception as e:  # noqa: BLE001 - keep the tool alive; show the error instead of dying in a thread
                self.emit(ev.Notice(f"{type(e).__name__}: {e}", "error", clip=True))
                if self.cfg.sounds:
                    play_cue("error")

    def handle(self, mode: ev.Mode, audio: np.ndarray) -> None:
        if len(audio) < SAMPLE_RATE // 4 or np.abs(audio).max() < 1e-3:
            self.emit(ev.Notice("too short or silent; check the terminal's Microphone permission if this repeats",
                                "warning", clip=True))
            return
        with self._gpu:
            result = self.engine.transcribe(audio, sentences=mode == "translate")
            text = postprocess.process(result.text, self.cfg, self.words)
            if mode == "translate" and text:
                try:
                    text = self.translate([postprocess.process(s, self.cfg, self.words) for s in result.sentences])
                except Exception as e:  # noqa: BLE001 - the Amharic text must not be lost with the translation
                    self.emit(ev.Notice(f"could not translate: {translate_problem(e)}; the Amharic text is in "
                                        "the history", "error"))
                    self.emit(ev.Transcribed("dictate", text, result.text, len(audio) / SAMPLE_RATE,
                                             result.compute_seconds, False))
                    return
        with self._paste_lock:
            if self._stopped.is_set():  # quitting: the ⌘V would land in the terminal the user just quit
                return
            pasted = bool(text) and paste.paste(text + " ", self.cfg.restore_delay)
        self.emit(ev.Transcribed(mode, text, result.text, len(audio) / SAMPLE_RATE, result.compute_seconds, pasted))

    def translate(self, sentences: list[str]) -> str:
        """Caller holds _gpu."""
        if self.translator is None:
            self.emit(ev.Loading("loading translation model"))
            self.translator = self._new_translator()
        return self.translator.translate(sentences, self.spec.language)

    def _new_translator(self):
        from qal_scribe.translate import Translator

        translator = Translator(self.engine.device)
        translator.load()
        return translator

    def _preload_translator(self) -> None:
        """Load the translator while the user is still speaking: 2.5 s that would otherwise follow the release."""
        with self._gpu:
            if self.translator is None and not self._stopped.is_set():
                with contextlib.suppress(Exception):  # the clip's own translate() tries again and reports it
                    self.translator = self._new_translator()


def translate_problem(e: Exception) -> str:
    from huggingface_hub.errors import LocalEntryNotFoundError

    if isinstance(e, LocalEntryNotFoundError):
        return "the translation model is not downloaded (`qal-scribe models download --translation` fetches it)"
    return f"{type(e).__name__}: {e}"
