from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from culinary_vision.vision.preprocess import PreprocessConfig, preprocess_frame


@dataclass
class FrameSignals:
    mean_hue: float
    mean_saturation: float
    mean_value: float
    brightness_std: float
    texture_laplacian: float
    edge_density: float
    foreground_ratio: float
    component_count: float
    mean_component_area: float
    flow_mean: float


class CulinaryFeatureExtractor:
    """Extract colour, texture, shape-proxy and motion signals from a pan ROI.

    These are observable visual signals, not direct measurements of food safety or
    internal doneness. A camera/ROI calibration and representative labels are needed
    before interpreting a state prediction operationally.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._previous_gray: Optional[np.ndarray] = None
        self._previous_summary: Optional[np.ndarray] = None

    @property
    def feature_names(self) -> list[str]:
        names = [
            "hsv_mean_h",
            "hsv_mean_s",
            "hsv_mean_v",
            "hsv_std_h",
            "hsv_std_s",
            "hsv_std_v",
            "lab_mean_l",
            "lab_mean_a",
            "lab_mean_b",
            "lab_std_l",
            "lab_std_a",
            "lab_std_b",
        ]
        names += [f"hue_hist_{i}" for i in range(12)]
        names += [f"sat_hist_{i}" for i in range(8)]
        names += [
            "laplacian_variance",
            "gray_entropy",
            "edge_density",
            "foreground_ratio",
            "component_count_log",
            "mean_component_area_ratio",
            "mean_component_solidity",
            "flow_mean",
            "flow_std",
            "flow_p90",
            "moving_pixel_ratio",
            "delta_mean_hue",
            "delta_mean_saturation",
            "delta_mean_value",
            "delta_texture",
        ]
        return names

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1)
        probabilities = hist / max(float(hist.sum()), 1.0)
        probabilities = probabilities[probabilities > 0]
        return float(-(probabilities * np.log2(probabilities)).sum())

    @staticmethod
    def _shape_signals(gray: np.ndarray) -> tuple[float, float, float, float]:
        # Otsu foreground proxy. It is intentionally a coarse shape/structure signal;
        # a task-specific segmentation model should replace it for production use.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(mask.mean()) > 127:
            mask = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        total_area = float(h * w)
        areas = [float(cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= 4]
        if not areas:
            return float(mask.mean() / 255.0), 0.0, 0.0, 0.0
        solidities = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 4:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidities.append(area / max(hull_area, 1.0))
        return (
            float(mask.mean() / 255.0),
            float(np.log1p(len(areas))),
            float(np.mean(areas) / total_area),
            float(np.mean(solidities)) if solidities else 0.0,
        )

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        frame = preprocess_frame(bgr_frame, self.config)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hsv_mean = hsv.reshape(-1, 3).mean(axis=0) / np.array([179.0, 255.0, 255.0])
        hsv_std = hsv.reshape(-1, 3).std(axis=0) / np.array([90.0, 128.0, 128.0])
        lab_mean = lab.reshape(-1, 3).mean(axis=0) / 255.0
        lab_std = lab.reshape(-1, 3).std(axis=0) / 128.0
        hue_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).reshape(-1)
        sat_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).reshape(-1)
        hue_hist = hue_hist / max(float(hue_hist.sum()), 1.0)
        sat_hist = sat_hist / max(float(sat_hist.sum()), 1.0)

        laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0
        entropy = self._entropy(gray) / 5.0
        edges = cv2.Canny(gray, 70, 140)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        foreground_ratio, component_count, component_area, solidity = self._shape_signals(gray)

        flow_mean = flow_std = flow_p90 = moving_ratio = 0.0
        if self._previous_gray is not None and self._previous_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self._previous_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
            flow_mean = float(magnitude.mean()) / 10.0
            flow_std = float(magnitude.std()) / 10.0
            flow_p90 = float(np.percentile(magnitude, 90)) / 10.0
            moving_ratio = float(np.mean(magnitude > 1.0))

        summary = np.array([hsv_mean[0], hsv_mean[1], hsv_mean[2], laplacian])
        deltas = np.zeros(4, dtype=np.float32)
        if self._previous_summary is not None:
            deltas = np.abs(summary - self._previous_summary)
        self._previous_summary = summary
        self._previous_gray = gray

        vector = np.concatenate(
            [
                hsv_mean,
                hsv_std,
                lab_mean,
                lab_std,
                hue_hist,
                sat_hist,
                np.array(
                    [
                        laplacian,
                        entropy,
                        edge_density,
                        foreground_ratio,
                        component_count,
                        component_area,
                        solidity,
                        flow_mean,
                        flow_std,
                        flow_p90,
                        moving_ratio,
                    ]
                ),
                deltas,
            ]
        ).astype(np.float32)
        if len(vector) != len(self.feature_names):
            raise RuntimeError(
                f"feature schema mismatch: {len(vector)} values vs {len(self.feature_names)} names"
            )
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_summary = None

    def signals(self, bgr_frame: np.ndarray) -> FrameSignals:
        """Return a compact human-readable signal summary for dashboards/logging."""
        vector = self.extract(bgr_frame)
        idx = {name: i for i, name in enumerate(self.feature_names)}
        return FrameSignals(
            mean_hue=float(vector[idx["hsv_mean_h"]]),
            mean_saturation=float(vector[idx["hsv_mean_s"]]),
            mean_value=float(vector[idx["hsv_mean_v"]]),
            brightness_std=float(vector[idx["hsv_std_v"]]),
            texture_laplacian=float(vector[idx["laplacian_variance"]]),
            edge_density=float(vector[idx["edge_density"]]),
            foreground_ratio=float(vector[idx["foreground_ratio"]]),
            component_count=float(vector[idx["component_count_log"]]),
            mean_component_area=float(vector[idx["mean_component_area_ratio"]]),
            flow_mean=float(vector[idx["flow_mean"]]),
        )
