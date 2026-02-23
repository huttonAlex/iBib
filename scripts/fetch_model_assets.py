#!/usr/bin/env python3
"""Download pinned model assets using models/manifest.json."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_root(manifest_path: Path) -> Path:
    return manifest_path.parent.parent


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent)) as tmp:
        with urllib.request.urlopen(url) as resp:
            shutil.copyfileobj(resp, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def _needs_download(path: Path, sha256: str | None) -> bool:
    if not path.exists():
        return True
    if sha256:
        return _sha256(path) != sha256
    return False


def _download_entry(root: Path, entry: dict, base_url: str, force: bool) -> bool:
    rel_path = entry.get("path")
    if not rel_path:
        return True
    url_path = entry.get("url_path", rel_path)
    expected = entry.get("sha256")
    dest = Path(rel_path)
    if not dest.is_absolute():
        dest = root / dest

    if not force and not _needs_download(dest, expected):
        print(f"OK: {rel_path}")
        return True

    url = base_url.rstrip("/") + "/" + url_path.lstrip("/")
    print(f"Downloading {url} -> {rel_path}")
    try:
        _download(url, dest)
    except Exception as exc:
        print(f"FAILED: {rel_path} ({exc})")
        return False

    if expected:
        actual = _sha256(dest)
        if actual != expected:
            print(f"CHECKSUM FAIL: {rel_path}\n  expected {expected}\n  actual   {actual}")
            return False

    return True


def fetch(manifest_path: Path, base_url: str, force: bool = False) -> bool:
    data = json.loads(manifest_path.read_text())
    root = _resolve_root(manifest_path)
    ok = True

    for art in data.get("artifacts", []):
        if not _download_entry(root, art, base_url, force):
            ok = False

    for tok in data.get("tokenizers", []):
        optional = bool(tok.get("optional", False))
        if not _download_entry(root, tok, base_url, force):
            if optional:
                print(f"WARNING: Optional tokenizer unavailable: {tok.get('path')}")
            else:
                ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch pinned model assets")
    parser.add_argument(
        "--manifest",
        type=str,
        default="models/manifest.json",
        help="Path to manifest JSON (default: models/manifest.json)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("POINTCAM_MODEL_BASE_URL"),
        help="Base URL for model downloads (or set POINTCAM_MODEL_BASE_URL)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if checksums match",
    )
    args = parser.parse_args()

    if not args.base_url:
        print("ERROR: --base-url is required (or set POINTCAM_MODEL_BASE_URL)")
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    ok = fetch(manifest_path, args.base_url, force=args.force)
    if ok:
        print("Model assets ready.")
        return 0
    print("Model asset fetch failed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
