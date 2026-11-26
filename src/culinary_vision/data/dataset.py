from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from culinary_vision.data.annotations import SegmentAnnotation, load_annotations
from culinary_vision.vision.corruptions import corrupt_frame
from culinary_vision.vision.features import CulinaryFeatureExtractor
from culinary_vision.vision.preprocess import PreprocessConfig


def extract_segment_features(
    annotation: SegmentAnnotation,
    extractor: CulinaryFeatureExtractor,
    sequence_length: int,
    corruption: str = "clean",
    seed: int = 7,
) -> np.ndarray:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    cap = cv2.VideoCapture(str(annotation.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {annotation.video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata: {annotation.video_path}")
    duration = frame_count / fps
    start = min(annotation.start_sec, max(0.0, duration - 1.0 / fps))
    end = min(annotation.end_sec, duration)
    times = np.linspace(start, max(start, end - 1.0 / fps), sequence_length)
    features = []
    extractor.reset()
    try:
        for i, timestamp in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000.0))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"failed reading {annotation.video_path} at {timestamp:.2f}s"
                )
            frame = corrupt_frame(frame, corruption, seed=seed + i)
            features.append(extractor.extract(frame))
    finally:
        cap.release()
    return np.stack(features).astype(np.float32)


def build_dataset(
    annotations_csv: str | Path,
    output_npz: str | Path,
    sequence_length: int = 16,
    preprocess: Optional[PreprocessConfig] = None,
    corruption: str = "clean",
    seed: int = 7,
    class_names_override: Optional[list[str]] = None,
) -> dict:
    annotations = load_annotations(annotations_csv)
    extractor = CulinaryFeatureExtractor(preprocess)
    sequences, labels, groups, starts = [], [], [], []
    for index, annotation in enumerate(annotations, start=1):
        sequence = extract_segment_features(
            annotation,
            extractor,
            sequence_length=sequence_length,
            corruption=corruption,
            seed=seed + index * 101,
        )
        sequences.append(sequence)
        labels.append(annotation.stage)
        groups.append(annotation.video_id)
        starts.append(annotation.start_sec)
    class_names = list(class_names_override) if class_names_override is not None else sorted(set(labels))
    label_to_id = {name: i for i, name in enumerate(class_names)}
    unknown_labels = sorted(set(labels) - set(class_names))
    if unknown_labels:
        raise ValueError(f"annotations contain labels absent from model class list: {unknown_labels}")
    y = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    x = np.stack(sequences).astype(np.float32)
    group_array = np.asarray(groups, dtype=str)
    output = Path(output_npz).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=x,
        y=y,
        groups=group_array,
        starts=np.asarray(starts, dtype=np.float32),
        class_names=np.asarray(class_names, dtype=str),
        feature_names=np.asarray(extractor.feature_names, dtype=str),
    )
    return {
        "output": str(output),
        "samples": int(len(y)),
        "videos": int(len(set(groups))),
        "classes": class_names,
        "features": int(x.shape[-1]),
        "sequence_length": int(x.shape[1]),
    }
