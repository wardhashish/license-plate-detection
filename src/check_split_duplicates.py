"""Check for near-duplicate images across train/val/test splits.

Uses a simple perceptual average hash so it does not require extra packages.
Low Hamming distance across different splits is a warning for possible leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_ROOT = BASE_DIR / "data" / "yolo" / "images"
RESULTS_DIR = BASE_DIR / "results"


def average_hash(path: Path, size: int = 8) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    return pixels > pixels.mean()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.reshape(-1) != b.reshape(-1)))


def image_files(split: str) -> list[Path]:
    split_dir = IMAGE_ROOT / split
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(split_dir.glob(ext))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check near-duplicate images across splits")
    parser.add_argument("--max-distance", type=int, default=5, help="Average-hash Hamming distance warning cutoff")
    parser.add_argument("--output", default=str(RESULTS_DIR / "split_duplicate_report.json"))
    args = parser.parse_args()

    records = []
    for split in ("train", "val", "test"):
        for path in image_files(split):
            records.append({"split": split, "path": str(path), "hash": average_hash(path)})

    warnings = []
    for i, left in enumerate(records):
        for right in records[i + 1 :]:
            if left["split"] == right["split"]:
                continue
            dist = hamming(left["hash"], right["hash"])
            if dist <= args.max_distance:
                warnings.append(
                    {
                        "left_split": left["split"],
                        "left_path": left["path"],
                        "right_split": right["split"],
                        "right_path": right["path"],
                        "hamming_distance": dist,
                    }
                )

    report = {
        "hash": "8x8 average hash",
        "max_distance": args.max_distance,
        "images_checked": len(records),
        "cross_split_warnings": warnings,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Images checked: {len(records)}")
    print(f"Cross-split near-duplicate warnings: {len(warnings)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
