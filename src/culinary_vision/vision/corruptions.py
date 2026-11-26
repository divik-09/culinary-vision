from __future__ import annotations

import cv2
import numpy as np


def corrupt_frame(frame: np.ndarray, kind: str, seed: int = 7) -> np.ndarray:
    """Apply deterministic camera-domain stress perturbations for robustness tests."""
    rng = np.random.default_rng(seed)
    image = frame.copy()
    if kind == "clean":
        return image
    if kind == "dim":
        return np.clip(image.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    if kind == "bright":
        return np.clip(image.astype(np.float32) * 1.35 + 18, 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(image, (9, 9), 2.0)
    if kind == "low_resolution":
        h, w = image.shape[:2]
        tiny = cv2.resize(image, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_AREA)
        return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 25])
        if not ok:
            raise RuntimeError("JPEG stress transform failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG stress transform could not decode image")
        return decoded
    if kind == "occlusion":
        h, w = image.shape[:2]
        x0, y0 = int(w * 0.58), int(h * 0.15)
        x1, y1 = min(w, int(w * 0.88)), min(h, int(h * 0.58))
        image[y0:y1, x0:x1] = 0
        return image
    if kind == "sensor_noise":
        noise = rng.normal(0.0, 12.0, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported corruption: {kind}")


CORRUPTION_SUITE = ["clean", "dim", "bright", "blur", "low_resolution", "jpeg", "occlusion", "sensor_noise"]
