import numpy as np
import pytest

from culinary_vision.vision.corruptions import CORRUPTION_SUITE, corrupt_frame


def test_all_corruptions_preserve_shape_and_dtype():
    frame = np.full((96, 128, 3), 120, dtype=np.uint8)
    for kind in CORRUPTION_SUITE:
        result = corrupt_frame(frame, kind)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


def test_unknown_corruption_raises():
    with pytest.raises(ValueError):
        corrupt_frame(np.zeros((10, 10, 3), dtype=np.uint8), "fish-eye")
