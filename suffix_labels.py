"""Append label suffixes to segmented video filenames.

This script renames files under the NiAD training set so that:
  - Files in `.../Training/Accident` end with `_A` before the extension
  - Files in `.../Training/Normal` end with `_N` before the extension

It skips any file that already ends with the correct suffix.

Usage (PowerShell):
  python suffix_labels.py

You can override the roots with --accident and --normal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def iter_videos(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            yield p


def needs_suffix(name: str, target_suffix: str) -> bool:
    # name is stem without extension; ensure we don't double-add
    return not (name.endswith("_A") or name.endswith("_N"))


def rename_with_suffix(file_path: Path, target_suffix: str) -> Tuple[bool, Path]:
    stem = file_path.stem
    if not needs_suffix(stem, target_suffix):
        return False, file_path

    new_name = f"{stem}{target_suffix}{file_path.suffix}"
    new_path = file_path.with_name(new_name)
    file_path.rename(new_path)
    return True, new_path


def apply_suffix(root: Path, suffix: str) -> Tuple[int, int]:
    renamed = 0
    skipped = 0
    for vid in iter_videos(root):
        did, _ = rename_with_suffix(vid, suffix)
        if did:
            renamed += 1
        else:
            skipped += 1
    return renamed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append _A/_N to video names")
    parser.add_argument(
        "--accident",
        type=Path,
        default=Path(r"E:\coding\datasetPreparation\NiAD_Large Videos\Training\Accident"),
        help="Path to Accident directory",
    )
    parser.add_argument(
        "--normal",
        type=Path,
        default=Path(r"E:\coding\datasetPreparation\NiAD_Large Videos\Training\Normal"),
        help="Path to Normal directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.accident.exists():
        raise FileNotFoundError(f"Accident directory not found: {args.accident}")
    if not args.normal.exists():
        raise FileNotFoundError(f"Normal directory not found: {args.normal}")

    a_renamed, a_skipped = apply_suffix(args.accident, "_A")
    n_renamed, n_skipped = apply_suffix(args.normal, "_N")

    print(
        "\nDone!"
        f"\n  Accident: renamed {a_renamed}, skipped {a_skipped}"
        f"\n  Normal:   renamed {n_renamed}, skipped {n_skipped}"
    )


if __name__ == "__main__":
    main()


