# How to explain CulinaryVision in an ML/CV interview

## 30-second project explanation

“I built CulinaryVision, a temporal computer-vision prototype for understanding what is happening inside a cooking pan. Instead of classifying isolated food photos, it extracts colour, texture, shape and motion signals from a fixed pan region, groups them into short frame sequences, and uses a bidirectional GRU to classify the visible cooking stage. I also built a video-disjoint evaluation workflow, camera-condition stress tests, and an optional YOLO pipeline for fine-grained ingredient/preparation-style detection. The current repository is a training and evaluation framework; its model quality depends on the labelled cooking sessions supplied to it.”

## Why use video instead of classifying one frame?

A single frame can be ambiguous. A few bubbles may be a splash; a colour shift may be caused by exposure; a noodle may look rigid in one frame but soften over time. Temporal features such as optical flow and changes in colour/texture let the model use persistence and direction of change. The GRU receives a sequence rather than one independent image.

## Why combine image processing with a temporal neural model?

The engineered signals make the pipeline compact and interpretable: HSV/Lab statistics capture colour, histograms capture colour distribution, Laplacian variance/entropy/edges act as texture proxies, contours provide coarse shape/occupancy cues, and Farneback optical flow estimates motion. A GRU learns how those signals evolve. This is a practical baseline for limited data, not a claim that handcrafted features are universally better than end-to-end CNN/video transformers.

## Why a bidirectional GRU?

The input is a short, already-recorded segment. A bidirectional GRU can use context from both earlier and later frames within that segment. It is relatively lightweight compared with a large video transformer. For strict streaming inference with no future context, replace it with a causal GRU/TCN and re-evaluate latency and accuracy.

## How do you prevent data leakage?

All segments from one source recording share a `video_id`. The split occurs at the video level, not at the segment/frame level. The trainer requires every class to appear in multiple independent recordings so the class can occur on both sides of the split. The external evaluation/robustness workflow rejects test recordings that appeared in training.

## What does the robustness suite test?

It applies deterministic brightness shifts, blur, resolution reduction, JPEG compression, partial occlusion and sensor noise to held-out videos. The report compares macro-F1 and accuracy against clean footage. This measures sensitivity to controlled image perturbations; it does not fully represent real steam, condensation, glare, camera shake or unfamiliar cookware. Those require actual recorded challenge clips.

## What do you measure?

- Macro-F1, because rare cooking stages should not disappear behind a majority-class accuracy score.
- Per-class precision and recall to see which visual states are confused.
- Confusion matrix to inspect adjacent-stage confusions.
- Inference latency on the target hardware, including feature extraction and model execution.
- Robustness drops by condition, alongside the number of test segments/sessions per class.

## What are the limitations?

- The visual features are proxies; they do not measure temperature, moisture, microbial safety or internal doneness.
- The current temporal classifier is not trained until labelled videos are supplied.
- The optional detector is a training integration point; fine-grained ingredient and prep-style performance requires custom bounding-box annotations and evaluation.
- Otsu-derived foreground/shape signals are coarse. A production system may need a trained segmentation model for overlapping ingredients and pan contents.
- ONNX export does not equal robot deployment. Camera calibration, compute profiling, sensor synchronization, failure handling and hardware validation remain.

## Likely follow-up questions

**What happens when the model is uncertain?**

The inference path applies a confidence threshold and maps low-confidence predictions to `unknown`. In a robot, that should trigger a conservative fallback or human/sensor check rather than a risky action.

**How would you improve it next?**

First curate a session-diverse dataset and improve annotation consistency. Then compare the current feature+GRU baseline against a CNN frame encoder plus temporal head, a lightweight video transformer, and task-specific segmentation. Add recipe-conditioned state labels and fuse camera signals with temperature, time and actuator telemetry where available.

**How would you move toward edge deployment?**

Export the compact temporal model to ONNX, benchmark end-to-end latency/memory on the target board, quantize only after checking accuracy/calibration, and test under thermal throttling and long-running sessions. Use a causal temporal model if future frames are not available at decision time.
