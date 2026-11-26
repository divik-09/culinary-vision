from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO detector on annotated culinary objects")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML file")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/ingredient_detection")
    parser.add_argument("--name", default="culinary_yolo")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install optional detector dependencies: pip install -e '.[vision]'") from exc
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=15,
        cache=False,
    )
    print(f"Training finished. Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
