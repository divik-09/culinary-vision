from __future__ import annotations

import argparse
import json
from pathlib import Path

from culinary_vision.data.annotations import write_annotation_template
from culinary_vision.data.dataset import build_dataset
from culinary_vision.demo import run_demo
from culinary_vision.inference import run_video_inference
from culinary_vision.training.evaluate import evaluate_dataset
from culinary_vision.training.robustness import run_robustness_suite
from culinary_vision.training.train import train_model
from culinary_vision.vision.preprocess import PreprocessConfig


def _parse_roi(value: str | None):
    if value is None:
        return None
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="culinary-vision",
        description="Temporal culinary scene understanding and robustness testing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("make-template", help="create a segment annotation CSV for a video")
    template.add_argument("--video", required=True)
    template.add_argument("--out", required=True)
    template.add_argument("--segment-seconds", type=float, default=5.0)

    dataset = sub.add_parser("build-dataset", help="extract temporal feature sequences from labelled clips")
    dataset.add_argument("--annotations", required=True)
    dataset.add_argument("--out", required=True)
    dataset.add_argument("--seq-len", type=int, default=16)
    dataset.add_argument("--roi", type=_parse_roi, default=None)

    train = sub.add_parser("train", help="train a video-disjoint temporal GRU classifier")
    train.add_argument("--dataset", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--epochs", type=int, default=35)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--patience", type=int, default=7)

    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint on an extracted dataset")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", required=True)

    stress = sub.add_parser("robustness", help="evaluate camera corruptions using labelled video segments")
    stress.add_argument("--checkpoint", required=True)
    stress.add_argument("--annotations", required=True)
    stress.add_argument("--out-dir", required=True)
    stress.add_argument("--seq-len", type=int, default=16)
    stress.add_argument("--roi", type=_parse_roi, default=None)

    infer = sub.add_parser("infer", help="run temporal stage inference on a cooking video")
    infer.add_argument("--video", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--out-video", required=True)
    infer.add_argument("--out-json", required=True)
    infer.add_argument("--sample-fps", type=float, default=4.0)
    infer.add_argument("--roi", type=_parse_roi, default=None)
    infer.add_argument("--detector-weights", default=None)
    infer.add_argument("--threshold", type=float, default=None)

    demo = sub.add_parser("demo", help="run a synthetic pipeline smoke test (not model training)")
    demo.add_argument("--out-dir", default="artifacts/demo")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "make-template":
        segments = write_annotation_template(args.video, args.out, args.segment_seconds)
        result = {"annotation_template": str(Path(args.out).resolve()), "segments": segments}
    elif args.command == "build-dataset":
        result = build_dataset(
            args.annotations,
            args.out,
            sequence_length=args.seq_len,
            preprocess=PreprocessConfig(roi=args.roi),
        )
    elif args.command == "train":
        result = train_model(
            args.dataset,
            args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            patience=args.patience,
        )
    elif args.command == "evaluate":
        result = evaluate_dataset(args.checkpoint, args.dataset)
    elif args.command == "robustness":
        result = run_robustness_suite(
            args.checkpoint,
            args.annotations,
            args.out_dir,
            sequence_length=args.seq_len,
            preprocess=PreprocessConfig(roi=args.roi),
        )
    elif args.command == "infer":
        result = run_video_inference(
            args.video,
            args.checkpoint,
            args.out_video,
            args.out_json,
            sample_fps=args.sample_fps,
            roi=args.roi,
            detector_weights=args.detector_weights,
            confidence_threshold=args.threshold,
        )
    elif args.command == "demo":
        result = run_demo(args.out_dir)
    else:
        raise RuntimeError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
