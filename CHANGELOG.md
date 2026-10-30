# Changelog

## 1.0.2

### New
- **Updates.** qal-scribe now tells you when a newer version is out (it asks PyPI at most once a day; turn it off
  with `check_updates = false`), and `qal-scribe update` installs it. `qal-scribe doctor` shows it too.
  Upgrading from 1.0.1 or older still needs `uv tool upgrade qal-scribe` once.

## 1.0.1

### New
- **A setup screen on first start.** It checks the permissions, lets you turn translation on or off, shows how big
  each download is and fetches the models with a progress bar. Open it again with `d` or `qal-scribe setup`.
- **Settings and help inside the app.** Press `s` to change the common settings (saved straight to `config.toml`)
  and `?` for the keys. Pressing a key to choose a hotkey now just works.
- **Translation is optional.** Set `translation = false`, or start with `qal-scribe --no-translate`, and the
  2.5 GB translation model is never downloaded or loaded. Good for small machines.
- **A choice of translation model.** `translation_model = "nllb-1.3b"` is a little more accurate than the default
  `nllb-600m` and about twice as slow. See `docs/translation.md` for the measurements.
- **A recording indicator on macOS.** A small pill shows "Listening" while you hold the key and "Transcribing…"
  afterwards, even when the terminal is hidden. Turn it off with `indicator = false`.
- **A clearer window.** Problems come with a one-line fix and a link to the right System Settings pane; the
  history shows the full text of the selected row; small terminals get a compact layout; the theme you pick is
  remembered.
- **Mute with one key.** Press `v` to turn the start, stop and error sounds off or on; it is saved to
  `config.toml` (`sounds = false`).
- **New README and a `docs/` folder** with install, settings, permissions, translation, languages and a FAQ.

### Fixed
- The default translate key on a Mac is now Right Command (Apple keyboards have no Right Control). Linux keeps
  Right Control.
- Number, spelling and punctuation clean-up for Amharic no longer changes text in Tigrinya and the other languages.
- The error sound now plays when a recording fails.
- Several fixes to the instructions: Intel Macs are not supported, Linux needs a few build tools, and the install
  command must use Python 3.12.

## 1.0.0

First public release: push-to-talk dictation for Amharic, with translation to English, on macOS and Linux.
