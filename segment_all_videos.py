"""Segment all videos in the NiAD training set into 30-frame clips.

This utility walks through the training dataset, splits every video into
30-frame segments, and names each output clip according to the required
convention:

    original_video_name_<segment_number>_<A|N>.mp4

Only the `Accident` and `Normal` categories receive the `_A` / `_N`
suffix. Videos that are not stored inside either category are still
segmented, but their clips keep the simpler
`original_video_name_<segment_number>.mp4` naming scheme and are saved in
an `Unlabeled` directory so they can be reviewed manually later.

Usage (PowerShell):

    python segment_all_videos.py

Adjust the constants near the bottom of the script if the dataset lives
somewhere else or you prefer a different output directory.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def infer_label(video_path: Path) -> Tuple[Optional[str], str]:
    """Infer the class label from the video's parent folders.

    Args:
        video_path: Path to the video file.

    Returns:
        A tuple ``(label_code, label_name)`` where ``label_code`` is either
        ``"A"`` (accident), ``"N"`` (normal), or ``None`` when no label can be
        inferred. ``label_name`` is the descriptive directory name to use in
        the output tree (``"Accident"``, ``"Normal"``, or ``"Unlabeled"``).
    """

    for part in video_path.parents:
        name = part.name.lower()
        if "accident" in name:
            return "A", "Accident"
        if "normal" in name:
            return "N", "Normal"
    return None, "Unlabeled"


def sanitize_stem(stem: str) -> str:
    """Convert the video stem to a filesystem-friendly base name."""

    # Replace whitespace with underscores and remove unsupported characters.
    stem = stem.strip().replace(" ", "_")
    stem = re.sub(r"[^0-9A-Za-z_-]", "", stem)
    # Collapse repeated underscores for neatness.
    stem = re.sub(r"_+", "_", stem)
    return stem or "segment"


def segment_video(
    video_path: Path,
    output_dir: Path,
    segment_length: int = 30,
    label_code: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Split ``video_path`` into ``segment_length``-frame clips.

    Args:
        video_path: Source video.
        output_dir: Directory where clips are stored.
        segment_length: Frames per clip (default: 30).
        label_code: ``"A"`` or ``"N"`` for labeled data; ``None`` leaves the
            label suffix off.
        overwrite: When ``False``, existing clips are not regenerated.

    Returns:
        Number of segments created.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # handle 0 or NaN
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        print(f"[WARN] Invalid frame size for video: {video_path}")
        cap.release()
        return 0

    base_name = sanitize_stem(video_path.stem)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    segment_index = 1
    frame_counter = 0
    writer = None
    created_segments = 0

    def close_writer():
        nonlocal writer
        if writer is not None:
            writer.release()
            writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # Flush any remaining frames in the current segment.
            if frame_counter > 0 and writer is not None:
                close_writer()
                created_segments += 1
            break

        if frame_counter == 0:
            label_suffix = f"_{label_code}" if label_code else ""
            segment_name = f"{base_name}_{segment_index}{label_suffix}.mp4"
            segment_path = output_dir / segment_name

            if segment_path.exists() and not overwrite:
                # Skip writing this segment entirely by fast-forwarding
                # through the next ``segment_length`` frames.
                skipped_frames = 1
                while skipped_frames < segment_length:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    skipped_frames += 1
                segment_index += 1
                frame_counter = 0
                continue

            writer = cv2.VideoWriter(
                str(segment_path),
                fourcc,
                fps,
                (width, height),
            )
            if not writer.isOpened():
                print(f"[WARN] Could not create writer for {segment_path}")
                break

        writer.write(frame)
        frame_counter += 1

        if frame_counter == segment_length:
            close_writer()
            created_segments += 1
            segment_index += 1
            frame_counter = 0

    close_writer()
    cap.release()

    return created_segments


def process_training_set(
    source_root: Path,
    output_root: Path,
    segment_length: int = 30,
    overwrite: bool = False,
) -> None:
    """Walk the training directory, segmenting every video we find."""

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    processed = 0
    skipped = 0

    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        label_code, label_dir = infer_label(path)
        destination = output_root / label_dir
        destination.mkdir(parents=True, exist_ok=True)

        segments_created = segment_video(
            video_path=path,
            output_dir=destination,
            segment_length=segment_length,
            label_code=label_code,
            overwrite=overwrite,
        )

        if segments_created:
            print(
                f"[INFO] {segments_created} segment(s) created for "
                f"{path.name} → {destination}"
            )
            processed += 1
        else:
            skipped += 1

    print(
        "\nDone!"
        f"\n  Videos processed: {processed}"
        f"\n  Videos skipped:   {skipped}"
        f"\n  Output root:      {output_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment NiAD training videos into 30-frame clips."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            r"E:\coding\datasetPreparation\NiAD_Large Videos\Training"
        ),
        help="Root directory containing the training videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"E:\coding\datasetPreparation\NiAD_Large Videos\Training_segments"
        ),
        help="Where to store the segmented clips.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=30,
        help="Number of frames per segment (default: 30).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate segments even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_training_set(
        source_root=args.source,
        output_root=args.output,
        segment_length=args.segment_length,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()


