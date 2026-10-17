import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import config

URL = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
DEST = config.DATA_DIR
TMP = config.ROOT / "data" / "_tmp"


def _extract_zip(zf, target):
    for member in zf.namelist():
        if member.startswith("__MACOSX/"):
            continue
        zf.extract(member, target)


def _find_dataset_root(root):
    for path in root.rglob("dataset-resized"):
        if path.is_dir() and any(p.is_dir() for p in path.iterdir()):
            return path
    return None


def main():
    if DEST.exists() and any(p.is_dir() for p in DEST.iterdir()):
        print(f"Dataset already present at {DEST}")
        return

    if not TMP.exists():
        print(f"Downloading TrashNet from {URL} ...")
        with urllib.request.urlopen(URL) as resp:
            data = resp.read()
        print("Extracting ...")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _extract_zip(zf, TMP)

    dataset_root = _find_dataset_root(TMP)
    if dataset_root is None:
        for nested in TMP.rglob("dataset-resized.zip"):
            print(f"Extracting nested {nested.name} ...")
            with zipfile.ZipFile(nested) as zf:
                _extract_zip(zf, TMP)
            dataset_root = _find_dataset_root(TMP)
            if dataset_root:
                break

    if dataset_root is None:
        raise SystemExit(f"Could not locate dataset-resized under {TMP}")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        shutil.rmtree(DEST, ignore_errors=True)
    shutil.move(str(dataset_root), str(DEST))
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"Dataset ready at {DEST}")


if __name__ == "__main__":
    main()
