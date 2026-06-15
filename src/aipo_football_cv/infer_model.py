from __future__ import annotations

from pathlib import Path

import pandas as pd
import supervision as sv
from ultralytics import YOLO

from .video import iter_video_frames


def infer_frames(
    model_path: str | Path,
    video_path: str | Path,
    output_csv: str | Path = "outputs/model_detections.csv",
    confidence: float = 0.3,
    max_frames: int | None = 200,
) -> pd.DataFrame:
    """Run only the trained YOLO model and save detections to CSV.

    This is useful for checking what the model learned before running the full
    tracking/statistics pipeline.
    """
    model = YOLO(str(model_path))
    rows = []
    for frame_id, frame in enumerate(iter_video_frames(video_path)):
        if max_frames is not None and frame_id >= max_frames:
            break
        result = model.predict(frame, conf=confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        names = {int(k): v for k, v in result.names.items()}
        if detections.xyxy is None:
            continue
        for bbox, class_id, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
            rows.append(
                {
                    "frame_id": frame_id,
                    "class_id": int(class_id),
                    "class_name": names.get(int(class_id), str(class_id)),
                    "confidence": float(conf),
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                }
            )
    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
