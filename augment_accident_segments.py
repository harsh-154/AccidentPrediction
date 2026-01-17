"""Augment 30-frame accident segments with brightness, shift, and flip."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def increase_brightness(frame, value: int = 40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def horizontal_shift(frame, shift_px: int = -30):
    rows, cols = frame.shape[:2]
    matrix = np.float32([[1, 0, shift_px], [0, 1, 0]])
    shifted = cv2.warpAffine(frame, matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return shifted


def horizontal_flip(frame):
    return cv2.flip(frame, 1)


def collect_videos(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def augment_video(
    video_path: Path,
    output_dir: Path,
    brightness_delta: int = 40,
    shift_pixels: int = 30,
    overwrite: bool = False,
) -> int:
    base_name = video_path.stem
    variants = {
        "bright": output_dir / f"{base_name}_bright.mp4",
        "shiftleft": output_dir / f"{base_name}_shift_left.mp4",
        "flip": output_dir / f"{base_name}_flip.mp4",
    }

    if not overwrite and all(path.exists() for path in variants.values()):
        print(f"[SKIP] All augmentations exist for {video_path.name}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writers: Dict[str, cv2.VideoWriter] = {}

    for key, out_path in variants.items():
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[WARN] Could not create writer for {out_path}")
            cap.release()
            for w in writers.values():
                w.release()
            return 0
        writers[key] = writer

    created = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if "bright" in writers:
            writers["bright"].write(increase_brightness(frame, brightness_delta))
        if "shiftleft" in writers:
            writers["shiftleft"].write(horizontal_shift(frame, -abs(shift_pixels)))
        if "flip" in writers:
            writers["flip"].write(horizontal_flip(frame))

    cap.release()

    for key, writer in writers.items():
        writer.release()
        created += 1

    return created


def augment_accident_segments(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    brightness_delta: int = 40,
    shift_pixels: int = 30,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for video_path in collect_videos(source_dir):
        output_video_dir = output_dir / video_path.parent.relative_to(source_dir)
        output_video_dir.mkdir(parents=True, exist_ok=True)

        created = augment_video(
            video_path=video_path,
            output_dir=output_video_dir,
            brightness_delta=brightness_delta,
            shift_pixels=shift_pixels,
            overwrite=overwrite,
        )

        if created:
            print(
                f"[INFO] Generated {created} augmentations for {video_path.name}"
            )
            processed += 1
        else:
            skipped += 1

    print(
        "\nAugmentation complete!"
        f"\n  Videos processed: {processed}"
        f"\n  Videos skipped:   {skipped}"
        f"\n  Output root:      {output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment accident video segments with brightness, shift, and flip"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            r"E:\coding\datasetPreparation\NiAD_Large Videos\Training_segments\Accident"
        ),
        help="Directory containing accident segments to augment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"E:\coding\datasetPreparation\NiAD_Large Videos\Training_segments\Accident_augmented"
        ),
        help="Directory where augmented videos will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate augmentations even if output files exist.",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=40,
        help="Additional brightness amount to add in HSV space (default: 40).",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=30,
        help="Number of pixels to shift left (default: 30).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    augment_accident_segments(
        source_dir=args.source,
        output_dir=args.output,
        overwrite=args.overwrite,
        brightness_delta=args.brightness,
        shift_pixels=args.shift,
    )


if __name__ == "__main__":
    main()


