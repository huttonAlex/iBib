#!/usr/bin/env python3
"""Verify pinned model assets against models/manifest.json."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_root(manifest_path: Path) -> Path:
    # manifest is expected at repo_root/models/manifest.json
    return manifest_path.parent.parent


def verify_manifest(manifest_path: Path) -> bool:
    data = json.loads(manifest_path.read_text())
    root = _resolve_root(manifest_path)

    ok = True
    artifacts = data.get("artifacts", [])
    if not artifacts:
        print("No artifacts found in manifest.")
        return False

    for art in artifacts:
        rel_path = art.get("path")
        if not rel_path:
            print("Artifact missing 'path' field.")
            ok = False
            continue
        path = Path(rel_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            print(f"MISSING: {rel_path}")
            ok = False
            continue

        expected = art.get("sha256")
        if expected:
            actual = _sha256(path)
            if actual != expected:
                print(f"MISMATCH: {rel_path}\n  expected {expected}\n  actual   {actual}")
                ok = False

        expected_size = art.get("size_bytes")
        if expected_size is not None:
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                print(
                    f"SIZE MISMATCH: {rel_path}\n  expected {expected_size}\n  actual   {actual_size}"
                )
                ok = False

    for tok in data.get("tokenizers", []):
        rel_path = tok.get("path")
        if not rel_path:
            continue
        path = Path(rel_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            optional = bool(tok.get("optional", False))
            if optional:
                print(f"WARNING: Missing optional tokenizer: {rel_path}")
            else:
                print(f"MISSING TOKENIZER: {rel_path}")
                ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pinned model assets")
    parser.add_argument(
        "--manifest",
        type=str,
        default="models/manifest.json",
        help="Path to manifest JSON (default: models/manifest.json)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    ok = verify_manifest(manifest_path)
    if ok:
        print("Model assets verified.")
        return 0
    print("Model asset verification failed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
