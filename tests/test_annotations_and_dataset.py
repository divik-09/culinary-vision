import csv
from pathlib import Path

import cv2
import numpy as np

from culinary_vision.data.annotations import load_annotations, write_annotation_template
from culinary_vision.data.dataset import build_dataset


def _write_video(path: Path, tint: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 72))
    assert writer.isOpened()
    for i in range(32):
        frame = np.full((72, 96, 3), (tint, 80 + (i % 3) * 5, 140), dtype=np.uint8)
        cv2.circle(frame, (20 + i, 34), 10, (20, 180, 60), -1)
        writer.write(frame)
    writer.release()


def test_annotation_template_and_dataset_build(tmp_path):
    video_a = tmp_path / "clip_a.mp4"
    video_b = tmp_path / "clip_b.mp4"
    _write_video(video_a, 35)
    _write_video(video_b, 180)

    template = tmp_path / "template.csv"
    count = write_annotation_template(video_a, template, segment_seconds=2.0)
    assert count >= 1

    annotation_file = tmp_path / "labels.csv"
    with annotation_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_path", "start_sec", "end_sec", "stage"])
        writer.writerow(["session_a", str(video_a), 0, 2, "raw_or_unheated"])
        writer.writerow(["session_b", str(video_b), 0, 2, "cooked_or_ready"])
    annotations = load_annotations(annotation_file)
    assert len(annotations) == 2

    output = tmp_path / "features.npz"
    summary = build_dataset(annotation_file, output, sequence_length=4)
    data = np.load(output)
    assert summary["samples"] == 2
    assert data["X"].shape == (2, 4, 47)
    assert set(data["class_names"].tolist()) == {"raw_or_unheated", "cooked_or_ready"}
