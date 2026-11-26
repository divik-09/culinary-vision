from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from culinary_vision.models.ingredient_detector import IngredientDetector
from culinary_vision.training.evaluate import load_model
from culinary_vision.vision.features import CulinaryFeatureExtractor
from culinary_vision.vision.preprocess import PreprocessConfig


def run_video_inference(
    video_path: str | Path,
    checkpoint_path: str | Path,
    output_video: str | Path,
    output_json: str | Path,
    sample_fps: float = 4.0,
    roi: Optional[Sequence[int]] = None,
    detector_weights: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
) -> dict:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    model, checkpoint, device = load_model(checkpoint_path)
    threshold = float(
        confidence_threshold
        if confidence_threshold is not None
        else checkpoint.get("confidence_threshold", 0.55)
    )
    detector = IngredientDetector(detector_weights) if detector_weights else None
    feature_extractor = CulinaryFeatureExtractor(PreprocessConfig(roi=roi))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open input video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("input video has invalid fps or frame dimensions")

    output_video = Path(output_video).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create output video: {output_video}")

    step = max(1, int(round(source_fps / sample_fps)))
    sequence = deque(maxlen=int(checkpoint["sequence_length"]))
    recent_predictions = deque(maxlen=5)
    records = []
    frame_index = 0
    sampled_frames = 0
    inference_time = []
    current_label = "warming_up"
    current_confidence = 0.0
    class_names = list(checkpoint["class_names"])
    frame_detections = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                start_time = time.perf_counter()
                sequence.append(feature_extractor.extract(frame))
                sampled_frames += 1
                if len(sequence) == sequence.maxlen:
                    batch = torch.from_numpy(np.stack(sequence)[None, ...]).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(model(batch), dim=1)[0].cpu().numpy()
                    raw_index = int(np.argmax(probs))
                    raw_confidence = float(probs[raw_index])
                    raw_label = class_names[raw_index]
                    if raw_confidence < threshold:
                        raw_label = "unknown"
                    recent_predictions.append(raw_label)
                    current_label = Counter(recent_predictions).most_common(1)[0][0]
                    current_confidence = raw_confidence
                    if detector is not None:
                        frame_detections = detector.predict(frame)
                    inference_time.append(time.perf_counter() - start_time)
                    records.append(
                        {
                            "timestamp_sec": frame_index / source_fps,
                            "stage": current_label,
                            "raw_prediction": class_names[raw_index],
                            "confidence": raw_confidence,
                            "detections": frame_detections,
                        }
                    )

            display = frame.copy()
            cv2.rectangle(display, (12, 12), (min(width - 12, 560), 74), (20, 20, 20), -1)
            text = f"Stage: {current_label} | confidence: {current_confidence:.2f}"
            cv2.putText(display, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(
                display,
                "Visual estimate only - verify with labelled cooking data",
                (24, 63),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (210, 210, 210),
                1,
            )
            for detection in frame_detections:
                x1, y1, x2, y2 = [int(v) for v in detection["xyxy"]]
                cv2.rectangle(display, (x1, y1), (x2, y2), (60, 220, 120), 2)
                cv2.putText(
                    display,
                    f"{detection['label']} {detection['confidence']:.2f}",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (60, 220, 120),
                    1,
                )
            writer.write(display)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    average_latency_ms = float(np.mean(inference_time) * 1000.0) if inference_time else None
    summary = {
        "input_video": str(Path(video_path).resolve()),
        "output_video": str(output_video),
        "sampled_frames": sampled_frames,
        "predictions": len(records),
        "average_model_latency_ms": average_latency_ms,
        "model_device": str(device),
        "confidence_threshold": threshold,
        "warning": "Predicted visual stage is not a food-safety or internal-temperature measurement.",
        "timeline": records,
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
