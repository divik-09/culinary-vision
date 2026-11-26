from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from culinary_vision.models.temporal_gru import TemporalGRU
from culinary_vision.training.metrics import classification_metrics


def _group_split(groups: np.ndarray, labels: np.ndarray, seed: int = 42, val_fraction: float = 0.2):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            "at least 2 distinct video_id values are required for a video-disjoint split; "
            "use several independently recorded cooking clips to prevent frame leakage"
        )
    classes = np.unique(labels)
    for class_id in classes:
        supporting_groups = np.unique(groups[labels == class_id])
        if len(supporting_groups) < 2:
            raise ValueError(
                f"class {int(class_id)} appears in fewer than 2 independent videos; "
                "record that stage in multiple sessions so it can exist in both train and validation"
            )

    rng = np.random.default_rng(seed)
    val_count = max(2, int(round(len(unique_groups) * val_fraction)))
    val_count = min(val_count, len(unique_groups) - 1)
    overall_counts = np.bincount(labels.astype(int), minlength=int(classes.max()) + 1).astype(float)
    best_score = float("inf")
    best_val_groups = None
    # Approximate stratification at the video level without leaking neighbouring segments.
    for _ in range(3000):
        chosen = set(rng.choice(unique_groups, size=val_count, replace=False).tolist())
        val_mask = np.asarray([group in chosen for group in groups], dtype=bool)
        train_labels = labels[~val_mask].astype(int)
        val_labels = labels[val_mask].astype(int)
        train_counts = np.bincount(train_labels, minlength=len(overall_counts))
        val_counts = np.bincount(val_labels, minlength=len(overall_counts))
        if np.any(train_counts == 0) or np.any(val_counts == 0):
            continue
        distribution_error = np.mean(np.abs((val_counts / overall_counts) - val_fraction))
        sample_error = abs(len(val_labels) / max(len(labels), 1) - val_fraction)
        score = float(distribution_error + 0.15 * sample_error)
        if score < best_score:
            best_score = score
            best_val_groups = chosen
    if best_val_groups is None:
        raise ValueError(
            "could not create a video-disjoint split with every class represented in both sets; "
            "collect more independent cooking sessions for each stage"
        )
    val_mask = np.asarray([group in best_val_groups for group in groups], dtype=bool)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def train_model(
    dataset_npz: str | Path,
    output_dir: str | Path,
    epochs: int = 35,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    dropout: float = 0.25,
    patience: int = 7,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    data = np.load(dataset_npz, allow_pickle=False)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["groups"].astype(str)
    class_names = data["class_names"].astype(str).tolist()
    feature_names = data["feature_names"].astype(str).tolist()
    if x.ndim != 3 or len(x) != len(y):
        raise ValueError("dataset must contain X with shape [samples, time, features] and y")
    if len(np.unique(y)) < 2:
        raise ValueError("training requires at least two cooking-stage classes")
    train_idx, val_idx = _group_split(groups, y, seed=seed)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    train_loader = DataLoader(
        TensorDataset(x_tensor[train_idx], y_tensor[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x, val_y = x_tensor[val_idx], y[val_idx]

    num_classes = len(class_names)
    counts = np.bincount(y[train_idx], minlength=num_classes).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / max(float(class_weights.mean()), 1e-8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGRU(x.shape[-1], num_classes, hidden_size, dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "temporal_gru.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_y)
            seen += len(batch_y)

        model.eval()
        with torch.no_grad():
            logits = model(val_x.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
        metrics = classification_metrics(y[val_idx], predictions, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{epochs} | loss={row['train_loss']:.4f} "
            f"| val_acc={row['val_accuracy']:.3f} | val_macro_f1={row['val_macro_f1']:.3f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.cpu().state_dict(),
                    "input_size": int(x.shape[-1]),
                    "num_classes": num_classes,
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "sequence_length": int(x.shape[1]),
                    "class_names": class_names,
                    "feature_names": feature_names,
                    "confidence_threshold": 0.55,
                    "train_video_ids": sorted(np.unique(groups[train_idx]).tolist()),
                    "validation_video_ids": sorted(np.unique(groups[val_idx]).tolist()),
                },
                checkpoint_path,
            )
            model.to(device)
            best_metrics = metrics
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "device": str(device),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "train_videos": sorted(np.unique(groups[train_idx]).tolist()),
        "validation_videos": sorted(np.unique(groups[val_idx]).tolist()),
        "class_names": class_names,
        "note": "Validation is grouped by video_id; metrics are only meaningful with representative, correctly labelled cooking footage.",
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
