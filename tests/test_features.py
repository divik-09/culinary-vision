import numpy as np
import pytest

from culinary_vision.vision.features import CulinaryFeatureExtractor
from culinary_vision.vision.preprocess import PreprocessConfig, crop_roi, preprocess_frame


def test_feature_extractor_returns_stable_finite_vector():
    extractor = CulinaryFeatureExtractor()
    first = np.full((180, 240, 3), (40, 120, 180), dtype=np.uint8)
    second = first.copy()
    second[45:125, 60:170] = (80, 90, 160)
    v1 = extractor.extract(first)
    v2 = extractor.extract(second)
    assert v1.shape == (len(extractor.feature_names),)
    assert v2.shape == v1.shape
    assert np.isfinite(v1).all()
    assert np.isfinite(v2).all()
    assert len(extractor.feature_names) == 47
    assert np.any(np.abs(v1 - v2) > 1e-5)


def test_roi_validation():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    assert crop_roi(frame, [10, 10, 50, 40]).shape[:2] == (40, 50)
    with pytest.raises(ValueError):
        crop_roi(frame, [90, 90, 50, 50])


def test_preprocess_rejects_empty_frame():
    with pytest.raises(ValueError):
        preprocess_frame(np.array([], dtype=np.uint8), PreprocessConfig())
