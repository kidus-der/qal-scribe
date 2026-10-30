# Questions

**Does anything I say leave my computer?**
No. Recording, transcription and translation all run on your computer. qal-scribe goes online only to fetch
the models from Hugging Face, and sends no audio, text or usage data anywhere.

**Why is the download so big?**
The speech model is about 2.4 GB and its language model 0.5 GB. Translation adds 2.5 GB (more for the bigger
models). You download them once; after that qal-scribe works offline.

**My computer is slow or has little memory. What can I do?**
Turn translation off: press `s` and switch off "Translation to English", or set `translation = false`. The
translation model is then never downloaded or loaded. `qal-scribe --no-translate` does the same for one run.
Without an Apple or NVIDIA GPU everything runs on the processor: it works, but a 10 second clip takes seconds
instead of a fraction of one.

**The keys do nothing.**
On a Mac, your terminal needs Input Monitoring: see [permissions](permissions.md), and quit and reopen the
terminal after turning it on. On Wayland there are no global keys: bind `qal-scribe toggle` to a key in your
desktop settings.

**The text appears in the window but is not pasted.**
On a Mac, turn on Accessibility for your terminal. On Linux, install the text tool `qal-scribe doctor` names.
Meanwhile, press `c` to copy the last text, or select a history row and press `Enter`.

**It pasted what I had copied before, not what I said.**
Some apps read the clipboard late. Raise `restore_delay` in `config.toml` (for example to `1.0`).

**Translation leaves sentences out or runs them together.**
qal-scribe cuts sentences where you pause for about half a second. Pause briefly between sentences. A bigger
model (`nllb-1.3b`) may translate better, at the cost of speed and memory.

**Why does it ask for Python 3.12?**
The language-model decoder that makes the text more accurate has no builds for newer Pythons. On another
version qal-scribe still works, with slightly less accurate decoding. uv downloads 3.12 for you.

**Does it work on Intel Macs or Windows?**
No. PyTorch no longer builds for Intel Macs, and Windows is not supported. Linux works and is in beta.

**Can I use it at work, or in a product?**
The code is MIT licensed. The Amharic speech model is CC-BY-4.0. The NLLB translation models are
for **non-commercial use only**. `qal-scribe about` lists every licence.

**Can it dictate other languages?**
Tigrinya, Afaan Oromo, Somali, Sidaama and Wolaytta are experimental, rough drafts at best. See
[languages](languages.md).

**How do I fix words it always gets wrong?**
Add them to `words.txt` as `heard => wanted`: see [your own words](settings.md#your-own-words).

**Something else is wrong.**
Run `qal-scribe doctor` and follow the fix it prints. If that does not help, please
[open an issue](https://github.com/kidus-der/qal-scribe/issues) with its output.
