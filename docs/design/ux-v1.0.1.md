# qal-scribe v1.0.1: TUI and first-run UX spec

Status: proposal, for one engineer in one pass. Audit of v1.0.0 (commit 1aa6d89) by rendering the Textual app with
the test `FakeController` at 120x36, 80x24, 80x14, 200x50 and compact 80x5 / 40x3, dark and light theme.
Screenshots cited below are in [`ux-v1.0.1/`](ux-v1.0.1/); regenerate all 38 states with
`XDG_CONFIG_HOME=$(mktemp -d) uv run python docs/design/ux-v1.0.1/shoot.py /tmp/shots` (no model, no audio).

Assumes these land first (other branches): `Config.translation: bool` and `Config.translation_model: str`
(default `"nllb-600m"`) with a table of translation models and their sizes (`feat/translation-models`); the macOS
floating recording indicator (`feat/recording-indicator`); the platform-default translate key (`fix/regressions`).
Rebase on them; nothing here changes their behaviour.

Ground rules: no rewrite of the main window, the controller or the event flow; no new dependencies (Textual's
`Switch`, `Select`, `Button`, `ModalScreen`, `Screen` and stdlib cover everything); new screens live in one new
module `src/qal_scribe/tui/screens.py`; every item below has its test listed.

## What works and stays

The state pill + stage border colour per state, the scrolling meter, live draft, history with copy-on-Enter,
compact mode, the two themes and the torch-free import are good. Recording, transcribing, translating and result
states read well at every size (`history-120.svg`). Do not restyle them.

## Gaps found

| # | Gap | Screenshot |
|---|---|---|
| G1 | First run downloads ~3 GB (+2.5 GB translation) in the plain terminal *before* the TUI opens (`cli.cmd_run` -> `fetch_missing_models`), with hub progress bars and no choice about translation. The README's way to skip translation is "edit `translate_key` before the first start". | (none: happens before the app) |
| G2 | Missing macOS permissions show as long, centred, wrapped warning paragraphs above "Hold Right Option to dictate", which still promises hotkeys that cannot work. No way to open the right System Settings pane; "add your terminal app" does not name the terminal. | `macperms-80.svg` |
| G3 | Compact mode cuts setup warnings off mid-sentence with no way to see the rest. | `compact-perms-40x3.svg` |
| G4 | Start failure: the error is shown twice (stage + 10 s toast), offers no fix, and the footer still offers Copy / Language / Compact, which do nothing. | `failed-80.svg` |
| G5 | Wayland: the hotkey warning and the hero say the same thing twice; "text insertion needs wtype" gives no install command. | `wayland-80.svg` |
| G6 | No settings UI: every option means finding and hand-editing `config.toml`; `t` changes the theme but the next start forgets it. | (all) |
| G7 | `doctor` fixes are prose ("System Settings > Privacy & Security > ..."), not commands; the PortAudio fix says `apt` on every distro and on macOS; Linux has no hotkey row at all (Wayland / no X display goes unreported); paste on Linux names packages but not the command. | `qal-scribe doctor` |
| G8 | Hero text is hard-coded: "dictate **Amharic**" after `l` switched to Tigrinya (`app.py` `hero()` `verbs`); key labels are raw pynput names (`Hold f13`), and "Right Option" is shown on Linux too (`hotkey.KEY_NAMES`). | `f13-120.svg` |
| G9 | Terminals under 18 rows: the stage keeps `min-height: 9`, the history panel collapses to its two borders and shows no rows. | `short-80x14.svg` |
| G10 | "(nothing recognised)" is reported as "not pasted" with the warning-coloured ✗, which reads as a failure. | `empty-result-120.svg` |
| G11 | History rows clip at 2 lines with no way to read the rest except copying it. | `long-result-120.svg` |
| G12 | The footer's key reminder (`#keys`) uses `$text-disabled`: nearly invisible in `qal-light`. | `light-history-120.svg` |
| G13 | No help: the only key list is the footer, which hides the push-to-talk keys below 110 columns. | `ready-80` (shoot.py) |
| G14 | Status line never says whether translation is on or which model it uses. | `history-120.svg` |
| G15 | Loading: brand glyph twice (header + stage), title "Starting", no indication of which model is loading. | `loading-120.svg` |

## Spec, in build order

### P0-1. Shared plumbing (small, first)

1. **`config.save(path, changes: dict[str, object | None]) -> Config`** in `config.py` (update the Contract line).
   Line-based, keeps every comment: for each key, replace the first line matching `^\s*key\s*=`, else the
   commented template line `^#\s*key\s*=`, else append `key = value`. `None` removes/comments the line (TOML has no
   null). Values: `true`/`false`, numbers as-is, strings via `json.dumps` (valid TOML basic strings). A missing
   file starts from `default_config.toml`. Validate before writing: parse the new text with the same rules as
   `load()` (split `load` into `parse(text, path)` + `load`), then `doctor.config_problem(cfg)`; on a problem raise
   `ValueError` and leave the file untouched. Write to `config.toml.tmp` then `os.replace` (atomic).
   Tests: round-trip keeps comments; commented default gets uncommented in place; invalid value raises and the
   file is byte-identical; missing file is created from the template.
2. **`doctor.Check` gets `url: str = ""`** (a pane to open). Add `doctor.terminal_name()`: `TERM_PROGRAM` ->
   `{"Apple_Terminal": "Terminal", "iTerm.app": "iTerm2", "ghostty": "Ghostty", "WezTerm": "WezTerm",
   "vscode": "VS Code"}`, else "your terminal app". Add `doctor.linux_install(*packages) -> str`: reads
   `platform.freedesktop_os_release()` `ID`/`ID_LIKE` and returns `sudo apt install ...` (debian/ubuntu),
   `sudo dnf install ...` (fedora/rhel), `sudo pacman -S ...` (arch), else "install: ..." with the names; package
   names per family come from one dict (`portaudio`: libportaudio2 / portaudio / portaudio; `build`:
   build-essential python3-dev / gcc python3-devel kernel-headers / base-devel; `wl-clipboard`, `ydotool`, `wtype`,
   `xdotool`, `xclip` are the same name everywhere).
3. **Settings panes** (macOS), one dict in `doctor.py`:
   - Microphone `x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone`
   - Accessibility `x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility`
   - Input Monitoring `x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent`
4. **Download with progress**: `models.download(spec, with_lm, progress=None)` and the translation download get
   `progress: Callable[[int, int], None] | None` (bytes done, bytes total). Implementation: pass
   `tqdm_class=` to `snapshot_download`, a subclass of `huggingface_hub.utils.tqdm` writing to `io.StringIO()`
   whose `update()` calls `progress(self.n, self.total)` when `self.unit == "B"` (hub 2.x aggregates every file
   into one byte bar). Size before downloading: `snapshot_download(..., dry_run=True)` -> sum `file_size` where
   `will_download`. Move the exception list and one-line reason out of `cli.friendly_download_errors` into
   `models.download_problem(e) -> str | None` so the TUI reports the same sentence. Tests: progress callback sees
   monotone bytes with a fake tqdm driver; `download_problem` on each hub error class.

### P0-2. doctor: every failing check prints the exact fix (G7)

Fix column becomes a command when one exists, a settings path plus `open "<url>"` on macOS. Exactly:

| Check | Platform | Fix text |
|---|---|---|
| Microphone permission (denied) | macOS | `open "x-apple...Privacy_Microphone"`, turn on {terminal}, then quit and reopen {terminal} |
| Accessibility | macOS | `open "x-apple...Privacy_Accessibility"`, turn on {terminal}, quit and reopen it |
| Input Monitoring | macOS | `open "x-apple...Privacy_ListenEvent"`, turn on {terminal}, quit and reopen it |
| Microphone: PortAudio missing | Linux | `linux_install("portaudio")` |
| Microphone: PortAudio missing | macOS | `uv tool install --python 3.12 --reinstall qal-scribe` (the wheel bundles PortAudio) |
| Microphone: no audio server | Linux | `systemctl --user start pipewire pipewire-pulse` |
| Microphone: no input / no match | all | `qal-scribe` then `s` (Settings > Microphone), or the `mic = ...` line with the names listed |
| Text insertion | Linux Wayland | `linux_install("wtype")` (sway, Hyprland) or `linux_install("ydotool", "wl-clipboard")` + "start ydotoold" (GNOME, KDE) |
| Text insertion | Linux X11 | `linux_install("xdotool", "xclip")` |
| **Hotkeys (new row)** | Linux | Wayland: warn, "bind `qal-scribe toggle` and `qal-scribe toggle --translate` in your desktop's keyboard settings"; pynput import fails: fail, the error and "run in an X11 or Wayland session" |
| Config | all | the path and `qal-scribe` then `s`, or `qal-scribe config --init` when there is no file |
| Speech / Language / Translation model | all | `qal-scribe models download [--translation]`, or `qal-scribe setup` |
| Compute device (CPU warn) | all | "turn translation off to save memory: `qal-scribe` then `s`" |

Tests: one per row's branch, monkeypatching the probe (`sys.platform`, `freedesktop_os_release`, `TERM_PROGRAM`),
asserting the fix string contains the command / URL.

### P0-3. First-run setup screen (G1, G2, G5)

`SetupScreen(Screen)` in `tui/screens.py`. Shown:
- automatically at start when `setup_needed(cfg)`: a needed model is not cached (speech; LM when `cfg.lm`, the
  spec has one and flashlight imports; translation when `cfg.translation`), or on macOS a permission row fails, or
  the microphone row fails (PortAudio / no input). Linux text-insertion problems do **not** open it every start
  (clipboard fallback is a valid way to work); they stay a hero note (P2-1).
- by `qal-scribe setup` (new subcommand: runs the TUI with `QalApp(cfg, setup=True)`), and by `d` in the main
  window (footer label "Setup").

Flow change: in `cli.cmd_run`, the TUI path no longer calls `fetch_missing_models` (`--plain` and `transcribe`
keep it). `QalApp.on_mount` pushes `SetupScreen` when needed and starts `boot()` only when it is dismissed with
"continue"; otherwise unchanged.

Layout (one column, max width 76, centred; works at 80x24):

```
 ቃል qal-scribe  Setup                                                  SETUP
 ╭─ Permissions ──────────────────────────────────────────────────────────────╮
 │ ✓ Microphone        granted                                                │
 │ ✗ Accessibility     iTerm2 cannot paste yet          [ Open Settings ]     │
 │ ✗ Input Monitoring  hotkeys cannot hear the keyboard [ Open Settings ]     │
 │   After turning one on, quit iTerm2 (⌘Q) and reopen it.                    │
 ╰────────────────────────────────────────────────────────────────────────────╯
 ╭─ Models ───────────────────────────────────────────────────────────────────╮
 │ Speech + language model  hohe-asr-amharic   2.9 GB   ━━━━━━━━──────── 48%  │
 │ [x] Translation to English  NLLB 600M  2.5 GB  non-commercial licence      │
 │                                            [ Download ]  [ Continue ]      │
 ╰────────────────────────────────────────────────────────────────────────────╯
 enter Download  c Continue  q Quit
```

- **Permission rows** (macOS) reuse `doctor.check_microphone_permission`, `check_paste`, `check_input_monitoring`
  and are re-run every 1 s by `set_interval` (all three are cheap pyobjc calls, no torch). Button per failing row:
  first press calls the system request, which also adds the terminal to the list (`CGRequestListenEventAccess()`,
  `AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})`, AVCaptureDevice
  `requestAccessForMediaType_completionHandler_` when status is 0 "not asked"); later presses run
  `subprocess.run(["open", check.url])`. Input Monitoring only reports granted after a relaunch, so once its
  button was pressed the row reads "restart {terminal} to finish" instead of "✗".
- **Linux block** instead of permissions: rows for Microphone (PortAudio), Text insertion and Hotkeys from the same
  doctor checks; each failing row shows its `linux_install(...)` command in a selectable `Static` and `c` copies
  it (`App.copy_to_clipboard` + `paste.copy`, as `action_copy` does). A footnote shows the build tools line
  (`linux_install("build")`) for reinstalls, since evdev needs a compiler at install time and the app cannot
  start without it.
- **Translation question**: only shown while a model is still to download or when opened with `setup`. A
  `Checkbox`/`Switch` "Translation to English" (default from `cfg.translation`), a `Select` of
  `translation_model` values from the translation table, and the selected model's size (from `dry_run`, "…" until
  known, "size unknown, offline" on failure) and licence. Changing either writes `translation` /
  `translation_model` immediately with `config.save`.
- **Download** runs in a daemon `threading.Thread` (as `boot` does, so `q` is never held hostage; `run()`'s
  `os._exit` check also covers this thread) and posts `DownloadProgress(row, done, total)` messages; each model
  row gets its own bar using `widgets.progress_bar`, plus "x of y GB". Errors show `models.download_problem(e)` in
  the row with `[ Retry ]`. When everything needed is cached and no row fails, the screen dismisses itself; a
  `Continue` button always lets the user go on (dictation works without translation, permissions can follow).
- Tests (`tests/test_tui.py` with `drive()`): shown when a model is missing, not shown when all cached; permission
  row flips to ✓ when the monkeypatched check flips; button calls `open` with the URL (mock `subprocess.run`);
  unticking translation writes `translation = false` to a temp config and drops its row; progress messages move
  the bar; download error shows the reason and Retry; `cmd_run` no longer pre-downloads for the TUI; `setup`
  subcommand parses. Add `setup-80` and `setup-120` to `test_tui_screenshots.SCENES`.

### P1-1. Settings screen (G6)

`SettingsScreen(Screen)`, key `s` ("Settings"), `Esc` closes. Two-column rows (label + help line, control).
Every change is validated and written at once with `config.save` (no separate Save button); on `ValueError` the
row shows the message in `$error` and the control reverts. Each row carries a tag: **live** (applied to the
shared `cfg` object, which the controller reads per clip) or **restart** (badge "applies at restart"; when any
restart-tagged value changed, the screen's footer offers `r Restart now`).

| Setting | Control | Applies |
|---|---|---|
| Translation | Switch (shows the model size if it is not downloaded: "downloads 2.5 GB at restart") | restart |
| Translation model | Select from the translation table, with size | restart |
| Language model (`lm`) | Switch | restart |
| Numbers as digits / Punctuation (experimental) / Merge homophones | Switch each | live |
| Sounds / Live preview | Switch each | live |
| Microphone | Select: "System default" + input names from `sounddevice.query_devices()` (`max_input_channels > 0`), read in a thread worker; stores the name, `None` for default | restart |
| Dictate key / Translate key | Button "Right Option — change"; opens `KeyCapture` modal | restart |
| Theme | Select: qal, qal-light, then the built-in themes | live (sets `app.theme`) |
| Compact | Switch | live |

- **KeyCapture** modal: "Press the key you want for Dictate. Esc cancels." Starts a `pynput.keyboard.Listener`
  for one press; accepts only `keyboard.Key` members (the same rule as `config_problem`), rejects character keys
  with "pick a key you never type with, such as Right Option or F13", and the other mode's key. While it is open
  the app's own push-to-talk is paused: add `PushToTalk.paused: bool` (`_press` returns early when set). On
  Wayland, or when the listener cannot start, it shows a `Select` of common names instead (alt_r, ctrl_r, cmd_r,
  shift_r, alt_l, f13 to f20). Translate key also offers "Off".
- Restart: `QalApp.restart = True`, then the normal quit path; `run()` re-executes the same command
  (`os.execv(sys.executable, [sys.executable, "-m", "qal_scribe", *sys.argv[1:]])`) after the app exits cleanly.
- `t` (cycle theme) also persists the theme with `config.save` (one line; G6).
- Tests: each live switch changes `app.cfg` and the file; invalid write (monkeypatch `config_problem`) shows the
  error and leaves the file alone; mic list from a mocked `query_devices`; capture accepts `Key.f13`, rejects a
  char key and the other mode's key; `paused` blocks `_press`; restart flag leads to `execv` (mocked). Screenshot
  scenes `settings-80`, `settings-120`.

### P1-2. Errors that say what to do (G4)

- `failed` state: hero = "Could not start" + the problem + one fix line from a small mapping on the exception
  text: unknown/absent mic -> "press s to pick a microphone"; "already running" -> "another qal-scribe is
  running: quit it first"; anything else -> "run `qal-scribe doctor`". Keys: `s` Settings, `d` Setup, `q` Quit.
- No toast for `StartFailed` (it is already in the stage).
- Footer shows only what works now: implement `QalApp.check_action` returning `False` for `copy` with nothing to
  copy, `language` while loading/failed/one language installed, `compact` while failed (Textual then dims them).
- Tests: failed hero contains the fix line per case; no toast on start failure; `check_action` states.

### P2. Main-window polish

1. **Setup notes as a callout** (G2, G3, G5): left-aligned, one line per problem, `⚠` + short summary (first
   clause of the message, e.g. "Accessibility is off: cannot paste") + "press d to fix"; full text goes to the
   setup screen. When Input Monitoring is missing, the "Hold Right Option" lines are replaced by "Hotkeys are off
   until Input Monitoring is on". Compact shows one line: "⚠ 2 setup problems · d". On Wayland drop the hotkey
   warning from the notes, since the hero already says to bind `qal-scribe toggle`.
2. **Hero uses the real language** (G8): `f"dictate {language_name}"`, from `models.MODELS[self.language]`.
   Key labels: extend `hotkey.KEY_NAMES` per platform (`alt_r` "Right Alt" on Linux, `cmd_r` "Right Super",
   `shift_r` "Right Shift", `f13` "F13"). Coordinate with `fix/regressions`, which owns the default translate
   key, so both use the same label table. Also derive `cli.EPILOG` from the config defaults rather than the
   hard-coded key names.
3. **Short terminals** (G9): Textual applies one breakpoint class per axis (the last threshold <= height), and
   `-small` is used nowhere, so make it `VERTICAL_BREAKPOINTS = [(0, "-tiny"), (4, "-short"), (18, "-tall")]`;
   `Screen.-short #stage { min-height: 5; max-height: 7; }` plus the hero title hidden, placed *before* the
   `Screen.-compact` block in `qal.tcss` so compact (same specificity, later) still wins at 3-5 rows.
4. **Nothing recognised** (G10): stage subtitle "nothing to paste", history mark `·` in `$text-muted`, not ✗.
5. **History actions** (G11): highlighting a row shows its full text and stats in the stage, titled
   "History · 19:38:09"; a new result or `Esc` returns the stage to the latest. `a` copies the Amharic source
   (`entry.raw`) of a translated row. Footer hint "enter Copy" while the list has focus.
6. **Help overlay** (G13): `?` opens `HelpScreen(ModalScreen)`: the push-to-talk keys from `Ready.hotkeys` (or the
   `qal-scribe toggle` lines), then every binding with its description, then "`qal-scribe doctor` checks
   everything". Any key closes it.
7. **Contrast** (G12): `#keys` colour `$text-muted`.
8. **Status line** (G14): append "→EN nllb-600m" or "translation off" from `cfg.translation` /
   `cfg.translation_model`.
9. **Loading** (G15): stage title "Loading {language_name}", hide `#hero-title` while loading (the header already
   shows the brand).

Tests: one assertion each in `test_tui.py` (text on screen / class set / clipboard content), and the existing
screenshot test keeps passing at all four sizes.

## Out of scope (named so nobody builds them)

Persistent history across runs, notification log, mouse-only flows, in-app language download (`l` keeps cycling
installed models), editing `words.txt` in the TUI, Windows.

## Checklist

- [ ] P0-1 `config.save` (comment-preserving, validated, atomic) + `config.parse`
- [ ] P0-1 `Check.url`, `terminal_name()`, `linux_install()`, macOS pane URLs in `doctor.py`
- [ ] P0-1 `models.download(..., progress=)` via `tqdm_class`, `dry_run` sizes, `models.download_problem`
- [ ] P0-2 doctor: exact fix command/URL for every failing row; new Linux Hotkeys row; per-distro PortAudio
- [ ] P0-3 `SetupScreen`: live permission rows + Open Settings/request buttons + restart hint
- [ ] P0-3 Linux block with copyable install commands
- [ ] P0-3 translation on/off + model + size before downloading; per-model progress; Retry; Continue
- [ ] P0-3 `qal-scribe setup`, `d` key, TUI path stops pre-downloading in `cmd_run`
- [ ] P1-1 `SettingsScreen` with live/restart tags, validation errors inline, `r` restart via `execv`
- [ ] P1-1 mic picker from real input devices; `KeyCapture` with `PushToTalk.paused` and Wayland fallback
- [ ] P1-1 `t` persists theme
- [ ] P1-2 failed state with a fix line, no duplicate toast, `check_action` hides dead keys
- [ ] P2-1 setup notes as one-line callouts; compact summary; no false "Hold ..." when hotkeys are off
- [ ] P2-2 hero uses the current language; platform key labels (with fix/regressions); EPILOG from defaults
- [ ] P2-3 `-short` breakpoint for < 18 rows
- [ ] P2-4 "nothing to paste" instead of ✗
- [ ] P2-5 highlighted history row in the stage; `a` copies the Amharic source
- [ ] P2-6 `?` help overlay with the real keys
- [ ] P2-7/8/9 key-hint contrast, translation in status line, loading title
- [ ] Screenshot scenes for setup, settings, help at 80x24 and 120x36
