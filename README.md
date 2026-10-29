# qal-scribe

On-device push-to-talk dictation for Amharic. Hold a key, speak, let go: the text appears wherever your cursor is.

## Install

```sh
uv tool install qal-scribe
qal-scribe doctor      # checks permissions, microphone and models, and says how to fix what is missing
qal-scribe             # hold Right Option, speak, release
```

On Linux without an NVIDIA GPU, install the CPU build of PyTorch instead of the multi-GB CUDA one:

```sh
uv tool install qal-scribe --torch-backend cpu
```

Work in progress.
