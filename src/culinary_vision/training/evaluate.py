from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from culinary_vision.models.temporal_gru import TemporalGRU
from culinary_vision.training.metrics import classification_metrics


def load_model(checkpoint_path: str | Path, device: str | None = None):
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model = TemporalGRU(
        checkpoint["input_size"],
        checkpoint["num_classes"],
        checkpoint["hidden_size"],
        checkpoint["dropout"],
    ).to(device_obj)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint, device_obj


def evaluate_dataset(checkpoint_path: str | Path, dataset_npz: str | Path) -> dict:
    model, checkpoint, device = load_model(checkpoint_path)
    data = np.load(dataset_npz, allow_pickle=False)
    x = torch.from_numpy(data["X"].astype(np.float32)).to(device)
    y = data["y"].astype(np.int64)
    dataset_classes = data["class_names"].astype(str).tolist()
    if dataset_classes != list(checkpoint["class_names"]):
        raise ValueError("dataset class order differs from the checkpoint; rebuild with the checkpoint class list")
    test_groups = set(data["groups"].astype(str).tolist())
    training_groups = set(checkpoint.get("train_video_ids", []))
    overlap = sorted(test_groups & training_groups)
    if overlap:
        raise ValueError(f"evaluation videos overlap training videos: {overlap}")
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(y, predictions, checkpoint["num_classes"])
    metrics["class_names"] = checkpoint["class_names"]
    metrics["samples"] = int(len(y))
    metrics["video_groups"] = sorted(np.unique(data["groups"].astype(str)).tolist())
    return metrics
