"""Fetch a small labelled test sample into eval/data/<set>-<lang>-test as <id>.wav + <id>.txt (16 kHz mono).

uv run --with scipy python eval/fetch_language_clips.py fleurs om_et [--limit 100]   -> eval/data/fleurs-om-test
uv run --with scipy python eval/fetch_language_clips.py waxal tir [--limit 100]      -> eval/data/waxal-tir-test

fleurs: the first clips of the FLEURS test split (CC-BY-4.0); the tarball is streamed, so only those clips download.
waxal: clips spread evenly over the Google WAXAL test split (per-language CC-BY or CC-BY-SA, see the dataset card),
read through the datasets-server rows API (48 kHz mp3, resampled to 16 kHz). FLEURS has no Tigrinya, Sidaama or Wolaytta.
The reference text is lowercased because these models emit lowercase Latin script.
"""
import argparse
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import soundfile as sf

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{code}/"
WAXAL = ("https://datasets-server.huggingface.co/rows?dataset=google/WaxalNLP&config={lang}_asr&split=test"
         "&offset={offset}&length=1")
OUT = Path(__file__).parent / "data"


def fetch(url: str) -> bytes:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503) or attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))  # datasets-server rate limit / overload


def fleurs(code: str, limit: int) -> Path:
    base = FLEURS.format(code=code)
    # test.tsv columns: id, file name, raw text, normalised (lowercase, no punctuation) text, ...
    refs = {row[1]: row[3] for row in (line.split("\t") for line in fetch(base + "test.tsv").decode().splitlines())}
    folder = OUT / f"fleurs-{code.split('_')[0]}-test"
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    with urllib.request.urlopen(base + "audio/test.tar.gz", timeout=60) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name in refs and n < limit:
                stem = Path(name).stem
                (folder / f"{stem}.wav").write_bytes(tar.extractfile(member).read())
                (folder / f"{stem}.txt").write_text(refs[name], encoding="utf-8")
                n += 1
            if n >= limit:
                break
    return folder


def waxal_row(lang: str, offset: int) -> dict:
    return json.loads(fetch(WAXAL.format(lang=lang, offset=offset)))


def waxal(lang: str, limit: int) -> Path:
    from scipy.signal import resample_poly

    total = waxal_row(lang, 0)["num_rows_total"]
    offsets = [i * total // limit for i in range(min(limit, total))]
    folder = OUT / f"waxal-{lang}-test"
    folder.mkdir(parents=True, exist_ok=True)

    def one(offset: int) -> None:
        row = waxal_row(lang, offset)["rows"][0]["row"]
        if (folder / f"{row['id']}.txt").exists():  # resume after an interrupted run
            return
        audio, sr = sf.read(io.BytesIO(fetch(row["audio"][0]["src"])), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype("float32")
        sf.write(folder / f"{row['id']}.wav", audio, 16000)
        (folder / f"{row['id']}.txt").write_text(row["transcription"].lower(), encoding="utf-8")

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one, offsets))
    return folder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["fleurs", "waxal"])
    p.add_argument("code", help="fleurs: om_et, so_so; waxal: tir, sid, wal, orm")
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    folder = (fleurs if args.source == "fleurs" else waxal)(args.code, args.limit)
    print(f"{len(list(folder.glob('*.wav')))} clips in {folder}")


if __name__ == "__main__":
    main()
