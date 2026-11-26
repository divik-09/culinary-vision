# Resume entry — only use claims that match your completed work

## Once the pipeline is implemented and tested (current repository state)

**CulinaryVision — Temporal Cooking Scene Understanding | Python, OpenCV, PyTorch**

- Built an image/video pipeline extracting colour, texture, shape and optical-flow features from pan-region footage, with sequence generation for temporal model training.
- Implemented a bidirectional GRU training/evaluation workflow with video-disjoint validation, confidence-based abstention, and camera-corruption stress tests.

## After training on real annotated cooking sessions

Add measured values only after running the held-out evaluation:

- Trained a temporal cooking-stage classifier on **[N] labelled sessions**, achieving **[macro-F1] macro-F1** across **[K] visual stages**; evaluated robustness to lighting, blur, occlusion and low-resolution footage.
- Fine-tuned a YOLO detector for **[ingredient classes]**, achieving **[mAP50-95] mAP50-95** on unseen cooking sessions; profiled inference at **[X] ms/frame** on **[hardware]**.

Do not imply robot deployment, proprietary cooking data, food-safety measurement or a trained ingredient detector unless you have actually completed those pieces.
