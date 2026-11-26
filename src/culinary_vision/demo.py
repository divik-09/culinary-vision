from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from culinary_vision.vision.features import CulinaryFeatureExtractor


def create_synthetic_cooking_video(path: str | Path, seconds: int = 12, fps: int = 12) -> Path:
    """Create a synthetic pan clip for plumbing/smoke tests, not model training."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("could not initialize MP4 writer; check OpenCV video codec support")
    total_frames = seconds * fps
    try:
        for index in range(total_frames):
            progress = index / max(total_frames - 1, 1)
            frame = np.full((height, width, 3), (45, 48, 52), dtype=np.uint8)
            cv2.rectangle(frame, (0, 0), (width, 70), (35, 38, 42), -1)
            cv2.ellipse(frame, (320, 205), (225, 125), 0, 0, 360, (18, 18, 20), -1)
            cv2.ellipse(frame, (320, 198), (210, 111), 0, 0, 360, (80, 82, 84), 5)
            ingredient_b = int(40 + 100 * progress)
            ingredient_g = int(115 - 35 * progress)
            ingredient_r = int(70 + 50 * progress)
            # Ingredient pieces slowly change colour and break into smaller pieces.
            count = 9 + int(progress * 18)
            rng = np.random.default_rng(index // 3)
            for _ in range(count):
                x = int(rng.integers(180, 460))
                y = int(rng.integers(135, 265))
                radius = int(rng.integers(5, 15)) if progress < 0.55 else int(rng.integers(3, 8))
                cv2.circle(frame, (x, y), radius, (ingredient_b, ingredient_g, ingredient_r), -1)
            # Simulated bubbling is only a visual test pattern.
            if progress > 0.35:
                bubble_count = int(2 + 22 * min(1.0, (progress - 0.35) / 0.65))
                for bubble in range(bubble_count):
                    bx = 210 + ((index * 11 + bubble * 37) % 220)
                    by = 155 + ((index * 7 + bubble * 19) % 95)
                    radius = 2 + ((index + bubble) % 5)
                    cv2.circle(frame, (bx, by), radius, (175, 178, 178), 1)
            cv2.putText(
                frame,
                "SYNTHETIC PLUMBING DEMO - NOT REAL COOKING FOOTAGE",
                (18, 337),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()
    return output


def run_demo(output_dir: str | Path) -> dict:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    video_path = create_synthetic_cooking_video(output / "synthetic_pan_demo.mp4")
    cap = cv2.VideoCapture(str(video_path))
    extractor = CulinaryFeatureExtractor()
    csv_path = output / "demo_frame_features.csv"
    count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % 6 == 0:
                vector = extractor.extract(frame)
                if writer is None:
                    writer = csv.writer(handle)
                    writer.writerow(["frame_index", *extractor.feature_names])
                writer.writerow([count, *[float(value) for value in vector]])
            count += 1
    cap.release()
    report = {
        "video": str(video_path),
        "frames_read": count,
        "feature_rows": len(list(csv.reader(csv_path.open(encoding="utf-8")))) - 1,
        "feature_count": len(extractor.feature_names),
        "feature_csv": str(csv_path),
        "warning": "Synthetic video validates the image/video pipeline only; it is not real cooking data and must not be used to claim model accuracy.",
    }
    (output / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
