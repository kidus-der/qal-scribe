"""Download the Amharic FLEURS dev and test splits (google/fleurs, CC-BY-4.0) for eval/evaluate.py.

uv run python eval/fetch_fleurs.py [--splits dev test]
Writes eval/data/fleurs-am-{dev,test}/<clip>.wav (16 kHz mono) and <clip>.txt (the raw transcription, with
punctuation; evaluate.py normalises both sides). Uses the raw files of the dataset repo, not its loading script.
"""
import argparse
import csv
import io
import tarfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "google/fleurs"
CONFIG = "am_et"
OUT = Path(__file__).parent / "data"


def fetch(split: str) -> int:
    tsv = hf_hub_download(REPO, f"data/{CONFIG}/{split}.tsv", repo_type="dataset")
    tar = hf_hub_download(REPO, f"data/{CONFIG}/audio/{split}.tar.gz", repo_type="dataset")
    with open(tsv, encoding="utf-8", newline="") as f:
        # columns: id, file_name, raw_transcription, transcription, characters, num_samples, gender
        raw = {row[1]: row[2] for row in csv.reader(f, delimiter="\t") if len(row) >= 3}
    out = OUT / f"fleurs-am-{split}"
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    with tarfile.open(tar) as t:
        for member in t:  # streamed into memory: nothing from the archive is extracted to disk by name
            name = Path(member.name).name
            if not member.isfile() or name not in raw:
                continue
            audio, sr = sf.read(io.BytesIO(t.extractfile(member).read()), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == 16000, f"{name}: expected 16 kHz, got {sr}"
            stem = Path(name).stem
            sf.write(out / f"{stem}.wav", audio, sr, subtype="PCM_16")
            (out / f"{stem}.txt").write_text(raw[name], encoding="utf-8")
            done += 1
    assert done == len(raw), f"{split}: {done} clips written, {len(raw)} listed in the tsv"
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])
    for split in p.parse_args().splits:
        print(f"fleurs-am-{split}: {fetch(split)} clips")


if __name__ == "__main__":
    main()
