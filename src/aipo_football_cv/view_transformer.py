from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .tracker import Tracks


class ViewTransformer:
    """Optional homography-based image-to-pitch coordinate transformer.

    The original notebook used Roboflow's field-keypoint model from Colab. Here it is
    optional and local-friendly: pass ``roboflow_api_key`` or set ``ROBOFLOW_API_KEY``.
    If dependencies or API key are missing, the pipeline can skip this step.
    """

    def __init__(self, model_id: str = "football-field-detection-f07vi/14", roboflow_api_key: str | None = None):
        self.model_id = model_id
        self.api_key = roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")
        self.model = None
        self.config = None
        if self.api_key:
            try:
                from inference import get_model
                from sports.configs.soccer import SoccerPitchConfiguration

                self.model = get_model(model_id=model_id, api_key=self.api_key)
                self.config = SoccerPitchConfiguration()
            except Exception as exc:
                raise RuntimeError(
                    "Field transformation dependencies are unavailable. Install optional dependencies "
                    "or run with view_transform.enabled=false."
                ) from exc

    @staticmethod
    def transform_point(point: tuple[int, int] | list[int], source_points: np.ndarray, target_points: np.ndarray) -> list[int] | None:
        homography_matrix, _ = cv2.findHomography(source_points, target_points)
        if point is None or homography_matrix is None:
            return None
        point_h = np.array([point[0], point[1], 1], dtype=np.float32).reshape(3, 1)
        transformed_h = homography_matrix @ point_h
        transformed = (transformed_h[:2] / transformed_h[2]).reshape(2)
        return transformed.astype(int).tolist()

    def add_transformed_position_to_tracks(self, tracks: Tracks, frames: list[np.ndarray], confidence: float = 0.3, keypoint_confidence: float = 0.5) -> None:
        if self.model is None or self.config is None:
            raise RuntimeError("ViewTransformer was initialized without a field-detection model.")

        import supervision as sv

        for frame_num, frame in tqdm(list(enumerate(frames)), desc="Transforming pitch coordinates"):
            result = self.model.infer(frame, confidence=confidence)[0]
            key_points = sv.KeyPoints.from_inference(result)
            mask = key_points.confidence[0] > keypoint_confidence
            if mask.sum() < 4:
                continue
            frame_reference_points = key_points.xy[0][mask]
            pitch_reference_points = np.array(self.config.vertices)[mask]

            # Map frame/image points onto pitch-reference coordinates.
            for object_tracks in tracks.values():
                for track_id, track_info in object_tracks[frame_num].items():
                    position = track_info.get("position")
                    if position is not None:
                        track_info["position_transformed"] = self.transform_point(
                            position,
                            frame_reference_points,
                            pitch_reference_points,
                        )
