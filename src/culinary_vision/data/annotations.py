from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SegmentAnnotation:
    video_id: str
    video_path: Path
    start_sec: float
    end_sec: float
    stage: str


def load_annotations(csv_path: str | Path) -> list[SegmentAnnotation]:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"annotation CSV not found: {path}")
    required = {"video_id", "video_path", "start_sec", "end_sec", "stage"}
    rows: list[SegmentAnnotation] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"annotation CSV is missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            if not (row.get("stage") or "").strip():
                continue
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid timestamps on CSV line {line_no}") from exc
            if start < 0 or end <= start:
                raise ValueError(f"invalid interval on CSV line {line_no}: [{start}, {end}]")
            raw_video = Path(row["video_path"]).expanduser()
            video_path = raw_video if raw_video.is_absolute() else (path.parent / raw_video).resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"video from CSV line {line_no} not found: {video_path}")
            rows.append(
                SegmentAnnotation(
                    video_id=(row.get("video_id") or video_path.stem).strip(),
                    video_path=video_path,
                    start_sec=start,
                    end_sec=end,
                    stage=row["stage"].strip(),
                )
            )
    if not rows:
        raise ValueError("no labelled annotation rows found; fill the 'stage' column")
    return rows


def write_annotation_template(
    video_path: str | Path,
    output_csv: str | Path,
    segment_seconds: float = 5.0,
) -> int:
    video_path = Path(video_path).expanduser().resolve()
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError("video metadata does not contain a valid fps/frame count")
    duration = frame_count / fps
    output = Path(output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "video_path", "start_sec", "end_sec", "stage"])
        start = 0.0
        while start < duration:
            end = min(duration, start + segment_seconds)
            writer.writerow([video_path.stem, str(video_path), f"{start:.3f}", f"{end:.3f}", ""])
            start = end
            count += 1
    return count
