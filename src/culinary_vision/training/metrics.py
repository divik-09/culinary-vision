from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= truth < num_classes and 0 <= prediction < num_classes:
            matrix[truth, prediction] += 1
    f1_values = []
    per_class = {}
    for index in range(num_classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if matrix[index, :].sum() > 0:
            f1_values.append(f1)
        per_class[str(index)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(matrix[index, :].sum()),
        }
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
