# Ingredient and preparation-style detection

The optional detector is intended for fine-grained classes that generic object datasets may not distinguish. Example classes:

- `onion_diced`, `onion_sliced`, `onion_puree`
- `carrot_baton`, `carrot_diced`
- `potato_wedge`, `potato_cube`
- `bell_pepper_sliced`, `bell_pepper_diced`
- `chicken_strip`, `chicken_mince`, `chicken_chunk`
- `tomato_diced`, `tomato_puree`
- `rice`, `pasta`

Choose classes that your recipes and camera can consistently distinguish. Do not create labels for distinctions that are visually ambiguous in the footage.

## Image dataset protocol

1. Capture images from independent cooking sessions, not only screenshots from a single video.
2. Include varied sizes, cut styles, ingredient quantities, pan surfaces, camera angles, lighting and partial occlusion.
3. Draw tight bounding boxes around visible ingredient regions. If overlapping ingredients cannot be separated reliably, annotate the visible region using a consistent rule rather than guessing hidden boundaries.
4. Split train/validation/test by session or source video before exporting YOLO labels. Near-duplicate frames from one clip must not cross splits.
5. Record per-class image counts and inspect class imbalance. Add more examples to rare classes instead of relying only on augmentation.
6. Keep a small, manually reviewed challenge set for visually similar classes (for example, diced versus small-cut ingredients).

## Training and evaluation

Train with `scripts/train_detector.py`. Evaluate the best checkpoint on untouched sessions and report per-class precision, recall, AP50 and mAP50-95 from the Ultralytics validation output. Inspect false positives/negatives by class and scene condition. Do not use training-set metrics as a performance claim.

For segmentation tasks such as separating touching ingredients or estimating occupied pan area, extend this workflow with a YOLO segmentation model and polygon masks. The current adapter is a bounding-box detector, not an instance-segmentation model.
