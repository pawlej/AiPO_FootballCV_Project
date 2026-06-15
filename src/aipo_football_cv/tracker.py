from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from .geometry import get_center_bbox, get_foot_position, get_width_bbox

Tracks = dict[str, list[dict[int, dict[str, Any]]]]


class Tracker:
    """YOLO detector + ByteTrack tracker for players, referees and ball."""

    def __init__(self, model_path: str | Path, confidence: float = 0.3, batch_size: int = 20):
        self.model = YOLO(str(model_path))
        self.tracker = sv.ByteTrack()
        self.confidence = confidence
        self.batch_size = batch_size

    def detect_frames(self, frames: list[np.ndarray]) -> list[Any]:
        detections = []
        for i in tqdm(range(0, len(frames), self.batch_size), desc="Detecting objects"):
            batch = frames[i : i + self.batch_size]
            detections.extend(self.model.predict(batch, conf=self.confidence, verbose=False))
        return detections

    @staticmethod
    def _class_name_map(result: Any) -> tuple[dict[int, str], dict[str, int]]:
        names = result.names
        names = {int(k): v for k, v in names.items()}
        return names, {v: k for k, v in names.items()}

    def get_object_tracks(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> Tracks:
        if read_from_stub and stub_path is not None and Path(stub_path).exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        results = self.detect_frames(frames)
        tracks: Tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, result in tqdm(list(enumerate(results)), desc="Tracking objects"):
            cls_names, cls_names_inv = self._class_name_map(result)
            detections = sv.Detections.from_ultralytics(result)

            # Treat goalkeeper as player, because team assignment handles jersey color later.
            if "goalkeeper" in cls_names_inv and "player" in cls_names_inv and detections.class_id is not None:
                goalkeeper_id = cls_names_inv["goalkeeper"]
                player_id = cls_names_inv["player"]
                detections.class_id = np.where(detections.class_id == goalkeeper_id, player_id, detections.class_id)

            tracked = self.tracker.update_with_detections(detections)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            if tracked.xyxy is not None and tracked.class_id is not None and tracked.tracker_id is not None:
                for bbox, class_id, tracker_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
                    class_name = cls_names.get(int(class_id), "")
                    if class_name == "player":
                        tracks["players"][frame_num][int(tracker_id)] = {"bbox": bbox.astype(float).tolist()}
                    elif class_name == "referee":
                        tracks["referees"][frame_num][int(tracker_id)] = {"bbox": bbox.astype(float).tolist()}

            if detections.xyxy is not None and detections.class_id is not None:
                for bbox, class_id in zip(detections.xyxy, detections.class_id):
                    if cls_names.get(int(class_id), "") == "ball":
                        tracks["ball"][frame_num][1] = {"bbox": bbox.astype(float).tolist()}
                        break

        if stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    @staticmethod
    def ball_pos_interpolation(ball_tracks: list[dict[int, dict[str, Any]]]) -> list[dict[int, dict[str, Any]]]:
        ball_pos_list = [pos.get(1, {}).get("bbox", [np.nan, np.nan, np.nan, np.nan]) for pos in ball_tracks]
        df_ball = pd.DataFrame(ball_pos_list, columns=["x1", "y1", "x2", "y2"])
        df_ball = df_ball.interpolate().bfill().ffill()
        return [{1: {"bbox": row.tolist()}} for row in df_ball.to_numpy()]

    @staticmethod
    def add_position_to_tracks(tracks: Tracks) -> None:
        for object_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    bbox = track_info["bbox"]
                    position = get_center_bbox(bbox) if object_type == "ball" else get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]["position"] = position

    @staticmethod
    def draw_ellipse(frame: np.ndarray, bbox: list[float], color: tuple[int, int, int], track_id: int | None = None) -> np.ndarray:
        bbox_center = get_center_bbox(bbox)
        bbox_width = get_width_bbox(bbox)
        center_coords = (bbox_center[0], int(bbox[3]))
        cv2.ellipse(frame, center_coords, (int(bbox_width), int(bbox_width * 0.3)), 0, -45, 230, color, 2)
        if track_id is not None:
            p1 = (int(bbox[0] + 0.7 * bbox_width), int(bbox[3] + 5))
            p2 = (int(bbox[2] - 0.7 * bbox_width), int(bbox[3] + 35))
            cv2.rectangle(frame, p1, p2, color, -1)
            cv2.putText(frame, str(track_id), (p1[0] + 5, p1[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    @staticmethod
    def draw_triangle(frame: np.ndarray, bbox: list[float], color: tuple[int, int, int] = (250, 107, 70)) -> np.ndarray:
        center = get_center_bbox(bbox)
        points = np.array([[center[0] - 10, int(bbox[1] - 10)], [center[0] + 10, int(bbox[1] - 10)], [center[0], int(bbox[1])]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, frames: list[np.ndarray], tracks: Tracks) -> list[np.ndarray]:
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            for track_id, player in tracks["players"][frame_num].items():
                color = tuple(int(v) for v in player.get("team_color", (0, 0, 255)))
                self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    self.draw_triangle(frame, player["bbox"], (0, 0, 255))
            for referee in tracks["referees"][frame_num].values():
                self.draw_ellipse(frame, referee["bbox"], (240, 239, 1))
            for ball in tracks["ball"][frame_num].values():
                self.draw_triangle(frame, ball["bbox"])
            output_frames.append(frame)
        return output_frames
