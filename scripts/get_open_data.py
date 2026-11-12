"""Fetch the open matches the submission trains on, and check every file against its pinned SHA-256.

    python scripts/get_open_data.py

Metrica's three sample games (github.com/metrica-sports/sample-data, commit e706dd5) go to
outputs/metrica/sample-data/data, or $MTAH_OPEN_GAMES; the 14 DFL/IDSSE files (Bassek et al. 2025,
CC BY 4.0, Hugging Face mirror pysport/idsse-data at a pinned revision) go to outputs/idsse, or
$MTAH_DFL_GAMES. A file already there with the right hash is kept. About 450 MB. Nothing about the
competition match is fetched or needed here.
"""
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtah import dfl, opengames  # noqa: E402

METRICA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
METRICA_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/" + METRICA_COMMIT + "/data/{}"
METRICA_LFS_URL = "https://media.githubusercontent.com/media/metrica-sports/sample-data/" + METRICA_COMMIT + "/data/{}"
DFL_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/" + dfl.REVISION + "/{}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(urls: list[str], dest: Path, expected: str, digest=sha256) -> None:
    """The first of urls whose download matches expected (a Git LFS file's plain URL gives a pointer,
    and a file that is not in LFS has no media URL)."""
    if dest.is_file() and digest(dest) == expected:
        print(f"  kept     {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
        except urllib.error.HTTPError:
            continue
        if digest(part) == expected:
            part.replace(dest)
            print(f"  fetched  {dest.name}")
            return
    part.unlink(missing_ok=True)
    raise SystemExit(f"{dest.name}: no download matches its pinned SHA-256 ({', '.join(urls)})")


def metrica_folder() -> Path:
    try:
        return opengames.folder()  # $MTAH_OPEN_GAMES, or a default folder that already holds games
    except FileNotFoundError:
        return Path(os.environ.get(opengames.ENV) or ROOT / opengames.DEFAULT_FOLDERS[0])


if __name__ == "__main__":
    games = metrica_folder()
    print(f"Metrica's open games -> {games}")
    for name, expected in opengames.SHA256.items():
        fetch([METRICA_URL.format(name), METRICA_LFS_URL.format(name)], games / name, expected, opengames.digest)
    matches = Path(os.environ.get(dfl.ENV) or ROOT / dfl.DEFAULT_FOLDER)
    print(f"DFL open matches -> {matches}")
    for name, expected in dfl.SHA256.items():
        fetch([DFL_URL.format(name)], matches / name, expected)
    print("all open files present and matching their pinned SHA-256")
