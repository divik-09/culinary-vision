from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    clahe: bool = True
    denoise: bool = False
    gamma: float = 1.0
    image_width: int = 320
    roi: Optional[Sequence[int]] = None


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction; gamma=1 leaves the image unchanged."""
    if gamma <= 0:
        raise ValueError("gamma must be greater than zero")
    if abs(gamma - 1.0) < 1e-8:
        return image.copy()
    lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(
        np.uint8
    )
    return cv2.LUT(image, lut)


def crop_roi(image: np.ndarray, roi: Optional[Sequence[int]]) -> np.ndarray:
    if roi is None:
        return image
    if len(roi) != 4:
        raise ValueError("roi must contain x, y, width, height")
    x, y, width, height = (int(v) for v in roi)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("roi coordinates must be non-negative with positive dimensions")
    h, w = image.shape[:2]
    if x + width > w or y + height > h:
        raise ValueError(f"roi {tuple(roi)} exceeds image dimensions {(w, h)}")
    return image[y : y + height, x : x + width]


def preprocess_frame(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Crop, resize and normalize a BGR frame for stable feature extraction."""
    if image is None or image.size == 0:
        raise ValueError("received an empty image")
    frame = crop_roi(image, config.roi)
    height, width = frame.shape[:2]
    target_width = max(32, int(config.image_width))
    if width > target_width:
        scale = target_width / float(width)
        frame = cv2.resize(
            frame,
            (target_width, max(32, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    frame = apply_gamma(frame, config.gamma)
    if config.denoise:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
    if config.clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        frame = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return frame
