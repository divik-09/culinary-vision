from __future__ import annotations

import argparse
from pathlib import Path

import torch

from culinary_vision.training.evaluate import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained temporal GRU model to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    model, checkpoint, _ = load_model(args.checkpoint, device="cpu")
    model.eval()
    dummy = torch.zeros(1, checkpoint["sequence_length"], checkpoint["input_size"])
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["visual_sequence"],
        output_names=["stage_logits"],
        dynamic_axes={"visual_sequence": {0: "batch"}, "stage_logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported {output}")


if __name__ == "__main__":
    main()
