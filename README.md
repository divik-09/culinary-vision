# CulinaryVision — Temporal Understanding of Food While It Cooks

**A computer-vision portfolio project for autonomous cooking systems**

CulinaryVision analyses a fixed camera view of a cooking pan and turns a sequence of frames into visual signals and cooking-stage predictions. It is designed around the kinds of scene changes that matter in autonomous cooking: ingredient appearance, colour and texture evolution, bubbling, changing structure, and liquid reduction.

> **Data honesty:** this repository contains the implemented pipeline, training/evaluation code, test suite and a synthetic smoke-test video generator. It does **not** contain a trained culinary model, real cooking dataset, fine-tuned ingredient detector or robot deployment. Those require labelled cooking footage and (for hardware deployment) a target device. The synthetic demo validates software plumbing only; never use it to claim model accuracy.

## Why this problem

A cooking system cannot rely only on a recipe timer. It needs visual feedback about what is in the pan and how it is changing. This project therefore treats cooking as a **temporal scene-understanding problem**, not a single-image food classifier.

The pipeline covers:

1. **Image processing:** ROI extraction, resizing, gamma correction, optional denoising and CLAHE contrast normalization.
2. **Visual feature engineering:** HSV/Lab colour statistics, colour histograms, texture/edge measurements, shape/occupancy proxies and optical-flow motion signals.
3. **Ingredient/prep-style detection integration:** an optional YOLO adapter and fine-tuning script for custom classes such as diced/sliced onion, carrot batons, potato cubes, chicken strips, pasta and rice. Fine-grained culinary classes need custom labelled images; generic COCO weights are not represented as a cooking detector.
4. **Video understanding:** a bidirectional GRU learns temporal patterns over frame-level visual features and classifies cooking-stage segments.
5. **Model evaluation:** video-grouped train/validation splits, macro-F1, per-class precision/recall and confusion matrices.
6. **Real-world robustness testing:** a repeatable held-out-video suite for dim/bright exposure, blur, low resolution, JPEG artefacts, occlusion and sensor noise.
7. **Inference and deployment preparation:** annotated video output, prediction timeline, latency measurement and ONNX export for the temporal classifier.

## System architecture

```text
Cooking video / camera recording
             |
             v
      ROI + preprocessing
             |
             v
  Colour / texture / shape / motion features ----> Optional custom YOLO detector
             |                                               |
             v                                               v
       16-frame sequence -------------------------- Ingredient boxes/classes
             |
             v
      Temporal BiGRU classifier
             |
             v
  Cooking-stage + confidence + timeline
             |
             v
 Held-out evaluation / camera stress suite / ONNX export
```

The temporal stage labels are `raw_or_unheated`, `heating_or_sauteing`, `browning_or_caramelizing`, `boiling_or_active_bubbling`, `reducing_or_pan_drying`, `cooked_or_ready` and `unknown`. These are human-annotated **visual categories**. They do not establish internal temperature, food safety or true doneness.

## Quick start

Python 3.10+ is recommended.

```bash
git clone <your-repository-url>
cd CulinaryVision
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the smoke test:

```bash
culinary-vision demo --out-dir artifacts/demo
```

It creates a synthetic MP4 and a CSV of extracted image/video features. This checks that OpenCV decoding, image processing and feature extraction work in the local environment. It is not training data.

Run the tests:

```bash
pytest -q
```

## Train the cooking-stage model on real clips

### 1. Collect and label cooking sessions

Record independent clips with the camera fixed above or beside the pan. Include full state transitions and difficult cases: steam, splashes, stirring, utensils, glare, different pan types and lighting. See [`docs/data_collection_protocol.md`](docs/data_collection_protocol.md) for the capture plan and label definitions.

Create a segment template for each clip:

```bash
culinary-vision make-template \
  --video data/raw/session_01.mp4 \
  --out data/annotations/session_01.csv \
  --segment-seconds 4
```

Fill the `stage` column. Combine labelled rows into one CSV with this schema:

```csv
video_id,video_path,start_sec,end_sec,stage
session_01,data/raw/session_01.mp4,0,4,raw_or_unheated
session_01,data/raw/session_01.mp4,4,8,heating_or_sauteing
session_02,data/raw/session_02.mp4,0,4,raw_or_unheated
```

Use a distinct `video_id` for every independent recording. Do not put adjacent segments from one recording into both training and test sets.

### 2. Build the temporal dataset

```bash
culinary-vision build-dataset \
  --annotations data/annotations/annotations.csv \
  --out data/processed/cooking_sequences.npz \
  --seq-len 16 \
  --roi 80,40,500,300
```

Omit `--roi` to use the full frame. The ROI must use source-frame pixel coordinates and remain identical for training, inference and evaluation. The default four-second annotation segment and 16-step sequence imply roughly 4 sampled frames/second; keep the training window duration and inference sampling rate consistent.

### 3. Train the temporal model

```bash
culinary-vision train \
  --dataset data/processed/cooking_sequences.npz \
  --out-dir artifacts/temporal_model \
  --epochs 35
```

The trainer holds out entire `video_id` groups for validation. It saves `temporal_gru.pt` and `training_summary.json`. At least two independent videos are required for the split; in practice, record multiple sessions per class and use more than one validation session.

### 4. Evaluate on unseen sessions

Build a separate dataset from clips whose `video_id` values were not used for training:

```bash
culinary-vision build-dataset \
  --annotations data/annotations/heldout.csv \
  --out data/processed/heldout.npz \
  --seq-len 16 \
  --roi 80,40,500,300

culinary-vision evaluate \
  --checkpoint artifacts/temporal_model/temporal_gru.pt \
  --dataset data/processed/heldout.npz
```

The evaluator checks that test videos do not overlap training videos and that class ordering matches the checkpoint.

### 5. Run robustness tests

```bash
culinary-vision robustness \
  --checkpoint artifacts/temporal_model/temporal_gru.pt \
  --annotations data/annotations/heldout.csv \
  --out-dir artifacts/robustness \
  --seq-len 16 \
  --roi 80,40,500,300
```

The report compares clean performance against deterministic camera-domain perturbations. Run this only on held-out clips. Add real recordings with steam, condensation, glare and camera vibration to complement synthetic perturbations.

## Run video inference

```bash
culinary-vision infer \
  --video data/raw/new_cooking_session.mp4 \
  --checkpoint artifacts/temporal_model/temporal_gru.pt \
  --out-video artifacts/inference/annotated.mp4 \
  --out-json artifacts/inference/timeline.json \
  --sample-fps 4 \
  --roi 80,40,500,300
```

The output video overlays the current stage and confidence. The JSON includes timestamps, model confidence, predictions and average model-call latency. Predictions below the confidence threshold are mapped to `unknown`.

## Optional ingredient and preparation-style detector

The repository includes a YOLO adapter and training script. To install it:

```bash
pip install -e ".[vision]"
```

Annotate images with bounding boxes in CVAT or Label Studio and export in YOLO detection format. Define your actual class taxonomy in a copy of [`configs/ingredient_dataset.yaml.example`](configs/ingredient_dataset.yaml.example), then train:

```bash
python scripts/train_detector.py \
  --data configs/ingredient_dataset.yaml \
  --model yolo11n.pt \
  --epochs 80 \
  --imgsz 640
```

Use the best checkpoint with the temporal inference pipeline:

```bash
culinary-vision infer ... --detector-weights runs/ingredient_detection/culinary_yolo/weights/best.pt
```

Do not claim ingredient detection performance until you have trained and evaluated this detector on a properly split, representative culinary dataset. See [`docs/ingredient_detection.md`](docs/ingredient_detection.md).

## Export the temporal model

```bash
python scripts/export_onnx.py \
  --checkpoint artifacts/temporal_model/temporal_gru.pt \
  --out artifacts/temporal_model/temporal_gru.onnx
```

ONNX export is a deployment-preparation step only. A real robot deployment still requires target-hardware profiling, camera calibration, memory/power checks, sensor integration, fail-safe behaviour and on-device validation.

## Repository layout

```text
src/culinary_vision/
  vision/       preprocessing, visual features, camera corruptions
  data/         segment annotations and video feature dataset builder
  models/       temporal GRU and optional YOLO ingredient detector
  training/     training, evaluation metrics and robustness suite
  inference.py  video prediction and annotated output
  cli.py        command-line interface
scripts/        YOLO fine-tuning and ONNX export
configs/        preprocessing defaults and example detector dataset config
docs/           data collection, annotation and evaluation protocols
tests/          unit and integration tests
```

## Current implementation status

| Component | Status |
|---|---|
| Image preprocessing and 47-dimensional visual feature extraction | Implemented and tested |
| Video decoding and temporal sequence construction | Implemented and tested |
| Trainable bidirectional GRU cooking-stage model | Implemented; requires real annotated clips to train |
| Video-disjoint validation and per-class metrics | Implemented |
| Camera corruption robustness suite | Implemented; requires held-out real clips |
| Annotated video inference and confidence-based abstention | Implemented; requires a trained checkpoint |
| Custom YOLO ingredient/prep-style detector adapter and trainer | Integration implemented; requires custom annotations and fine-tuning |
| ONNX export | Implemented; requires a trained checkpoint and ONNX dependency |
| Target robot integration / deployment | Not implemented; no target robot hardware or proprietary cooking dataset available |

## Resume wording

Use the project on your resume **after you have collected and labelled real cooking clips, trained the model, and run the held-out evaluation**. Replace the bracketed values with your measured results:

**CulinaryVision — Temporal Cooking Scene Understanding | Python, OpenCV, PyTorch, YOLO, ONNX**

- Built an end-to-end computer-vision pipeline for cooking videos, extracting colour, texture, shape and optical-flow features and classifying temporal cooking stages with a bidirectional GRU.
- Developed a custom ingredient/prep-style detection workflow and evaluated model robustness across lighting, blur, occlusion and resolution shifts on held-out cooking sessions.
- Achieved **[macro-F1]** across **[N]** cooking sessions and measured **[X] ms/frame** inference latency on **[hardware]**.

Do not include the final metric bullet until you have actually trained and evaluated it. If you have only completed the current codebase and smoke tests, describe it as an implemented pipeline/prototype—not as a validated cooking model.
