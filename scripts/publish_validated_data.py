"""Publish reviewed staged files without overwriting any different existing file."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--destination", type=Path, default=Path("/data/mongo"))
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for source in args.files:
        source = source.resolve(strict=True)
        target = args.destination / source.name
        if source == target.resolve():
            raise ValueError("Source and destination must differ")
        sha = digest(source)
        if target.exists() and digest(target) != sha:
            raise ValueError(f"Refusing to overwrite different existing file: {target}")
        entries.append((source, target, sha))
    receipt = []
    for source, target, sha in entries:
        created = not target.exists()
        if created:
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        if digest(target) != sha:
            raise RuntimeError(f"Published hash mismatch: {target}")
        receipt.append({"source": str(source), "destination": str(target), "sha256": sha,
                        "bytes": target.stat().st_size, "created": created, "original_overwritten": False})
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
