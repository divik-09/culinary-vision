import numpy as np
import torch

from culinary_vision.models.temporal_gru import TemporalGRU
from culinary_vision.training.metrics import classification_metrics


def test_temporal_gru_output_shape():
    model = TemporalGRU(input_size=47, num_classes=4)
    output = model(torch.randn(3, 16, 47))
    assert output.shape == (3, 4)


def test_classification_metrics_confusion_matrix():
    result = classification_metrics(np.array([0, 1, 1, 2]), np.array([0, 0, 1, 2]), 3)
    assert result["accuracy"] == 0.75
    assert len(result["confusion_matrix"]) == 3
    assert result["per_class"]["1"]["support"] == 2


def test_training_runs_with_video_disjoint_groups(tmp_path):
    from culinary_vision.training.train import train_model

    rng = np.random.default_rng(4)
    sequences = []
    labels = []
    groups = []
    for video_index in range(6):
        for label in (0, 1):
            base = np.zeros((6, 47), dtype=np.float32)
            base[:, label * 4 : label * 4 + 4] = 2.0
            base += rng.normal(0, 0.08, base.shape).astype(np.float32)
            sequences.append(base)
            labels.append(label)
            groups.append(f"session_{video_index}")
    dataset = tmp_path / "toy_sequences.npz"
    np.savez_compressed(
        dataset,
        X=np.stack(sequences),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        class_names=np.asarray(["stage_a", "stage_b"]),
        feature_names=np.asarray([f"f{i}" for i in range(47)]),
    )
    result = train_model(
        dataset,
        tmp_path / "trained",
        epochs=2,
        batch_size=4,
        hidden_size=8,
        patience=2,
    )
    assert (tmp_path / "trained" / "temporal_gru.pt").exists()
    assert set(result["train_videos"]).isdisjoint(set(result["validation_videos"]))
