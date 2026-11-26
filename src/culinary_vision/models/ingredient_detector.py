from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class IngredientDetector:
    """Optional YOLO adapter for custom ingredient/prep-style detection weights.

    The repository does not pretend that generic COCO weights understand fine-grained
    culinary categories. Train with your own annotated ingredient dataset exported in
    YOLO format, then pass the resulting weights here.
    """

    def __init__(self, weights: str | Path, confidence: float = 0.35, device: str = ""):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO support requires the optional dependencies: pip install -e '.[vision]'"
            ) from exc
        if not Path(weights).exists():
            raise FileNotFoundError(f"detector weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.device = device or None

    def predict(self, bgr_frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=bgr_frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(float).tolist()
                class_id = int(box.cls[0].item())
                detections.append(
                    {
                        "label": str(names[class_id]),
                        "confidence": float(box.conf[0].item()),
                        "xyxy": xyxy,
                    }
                )
        return detections
