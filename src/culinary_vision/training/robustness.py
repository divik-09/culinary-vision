from __future__ import annotations

from pathlib import Path

from culinary_vision.data.dataset import build_dataset
from culinary_vision.training.evaluate import evaluate_dataset
from culinary_vision.vision.corruptions import CORRUPTION_SUITE
from culinary_vision.vision.preprocess import PreprocessConfig


def run_robustness_suite(
    checkpoint_path: str | Path,
    annotations_csv: str | Path,
    output_dir: str | Path,
    sequence_length: int = 16,
    preprocess: PreprocessConfig | None = None,
) -> dict:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from culinary_vision.data.annotations import load_annotations

    annotated = load_annotations(annotations_csv)
    test_groups = {item.video_id for item in annotated}
    training_groups = set(checkpoint.get("train_video_ids", []))
    overlap = sorted(test_groups & training_groups)
    if overlap:
        raise ValueError(f"robustness videos overlap model training videos: {overlap}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for corruption in CORRUPTION_SUITE:
        npz_path = output / f"stress_{corruption}.npz"
        build_dataset(
            annotations_csv,
            npz_path,
            sequence_length=sequence_length,
            preprocess=preprocess,
            corruption=corruption,
            class_names_override=list(checkpoint["class_names"]),
        )
        report[corruption] = evaluate_dataset(checkpoint_path, npz_path)
        print(
            f"{corruption:16s} accuracy={report[corruption]['accuracy']:.3f} "
            f"macro_f1={report[corruption]['macro_f1']:.3f}"
        )
    clean_f1 = report["clean"]["macro_f1"]
    for name, result in report.items():
        result["macro_f1_drop_vs_clean"] = float(clean_f1 - result["macro_f1"])
    import json

    target = output / "robustness_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report": report, "output": str(target)}
