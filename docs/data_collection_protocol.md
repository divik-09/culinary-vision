# Cooking-video data collection and annotation protocol

This project needs **your own labelled cooking footage** before any model-quality claim is valid. Do not use the synthetic demo clip for training or evaluation.

## Capture protocol

1. Mount a phone or webcam securely above or obliquely facing the pan. Keep the camera fixed for the whole clip.
2. Capture at 720p or higher and 10–30 FPS. Avoid digital zoom. Lock focus/exposure/white balance when the camera supports it; also record a few clips with auto-exposure enabled for stress testing.
3. Keep the pan/cooking area inside a consistent region of interest (ROI). Record the ROI as `x,y,width,height` in source-frame pixels and use the same ROI during dataset building, inference and evaluation.
4. Record complete sessions, not just ideal moments. Include ingredient addition, stirring, steam, splashes, utensil occlusion, reflections, lighting changes and the end of cooking.
5. Repeat the same recipe across different days, pans, cameras, lighting conditions, ingredient quantities and cut sizes. Give every independent recording a distinct `video_id`.
6. Do not film people or identifiable personal material unless they have consented. Store footage locally and remove it when no longer needed.

## Visual stage taxonomy

Use the following labels for the *dominant visible state in the pan ROI*. They are visual categories, not a food-safety determination.

| Label | Annotation guidance |
|---|---|
| `raw_or_unheated` | Ingredients are present before meaningful visible heating/transformation. |
| `heating_or_sauteing` | Ingredients are heating or moving in oil/liquid; no sustained browning/boiling cue dominates. |
| `browning_or_caramelizing` | Sustained visible golden/brown colour development is the key change. |
| `boiling_or_active_bubbling` | Repeated, sustained bubbles or surface agitation are visible. |
| `reducing_or_pan_drying` | Free liquid visibly decreases; the pan becomes drier or ingredients become less submerged. |
| `cooked_or_ready` | The annotated recipe stage is visually complete according to a human-defined criterion. |
| `unknown` | The view is blocked, ambiguous, between classes, or unsuitable for a confident label. |

For each segment, annotate the state that is visible for most of the interval. Use `unknown` instead of guessing. If one segment contains a clear transition, split it into shorter segments.

## Annotation workflow

Create a template for one clip:

```bash
culinary-vision make-template --video data/raw/session_01.mp4 --out data/annotations/session_01.csv --segment-seconds 4
```

Fill the `stage` column with one label from the taxonomy. Keep the same video ID for all segments from the same recording. Combine the rows into one `annotations.csv` for dataset preparation. Use the full absolute video path or a path relative to the CSV file.

## Split discipline

- Split by `video_id`, never by randomly splitting adjacent frames or segments. Adjacent frames from the same cooking session are highly correlated and would inflate test scores.
- Aim for at least 5–10 independent sessions per important class before relying on metrics. More varied sessions are better than many segments from one long clip.
- Keep a separate set of unseen sessions for the final robustness suite. The code rejects overlap with training video IDs.
- Report macro-F1, per-class precision/recall, confusion matrix, class support and the exact test sessions. Include uncertainty and failure cases when sample counts are small.

## Real-world test matrix

Evaluate held-out clips in at least these conditions:

- Normal lighting and fixed camera.
- Dim and bright exposure.
- Steam, condensation and partial utensil/hand occlusion.
- Motion blur, camera shake and low resolution.
- Reflective cookware and visually similar ingredient colours.
- Different pan types, camera angles and ingredient quantities.
- Long clips with state transitions and pauses between actions.

The automated suite currently includes dim/bright exposure, Gaussian blur, low resolution, JPEG compression, synthetic occlusion and sensor noise. It is a repeatable stress test, not a substitute for recording actual steam, glare, camera vibration or new kitchens.

## Important limitations

- Colour/texture/shape features are image-derived proxies. They do not directly measure temperature, moisture, ingredient identity or food safety.
- A task-specific detector must be trained on annotated culinary images to distinguish fine-grained ingredient variants and prep styles. The optional YOLO adapter is only an integration point until those weights exist.
- The compact GRU requires representative labelled cooking sessions. Do not report accuracy from the synthetic smoke-test video.
- Real robot deployment additionally requires camera calibration, hardware-specific latency/power tests, sensor integration, fail-safe behaviour and human sign-off. This repository exports the temporal model to ONNX but does not claim deployment on a physical robot.
