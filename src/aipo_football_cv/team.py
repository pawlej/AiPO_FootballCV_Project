from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """Assign players to two teams using dominant jersey colors and KMeans."""

    def __init__(self):
        self.team_colors: dict[int, tuple[int, int, int]] = {}
        self.player_team_dict: dict[int, int] = {}
        self.kmeans: KMeans | None = None

    @staticmethod
    def _clip_bbox(frame: np.ndarray, bbox: list[float]) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(bbox[0])))
        y1 = max(0, min(h - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(w, int(bbox[2])))
        y2 = max(y1 + 1, min(h, int(bbox[3])))
        return x1, y1, x2, y2

    @staticmethod
    def get_clustering_model(image: np.ndarray) -> KMeans:
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=42)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        x1, y1, x2, y2 = self._clip_bbox(frame, bbox)
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        top_half = image[: max(1, image.shape[0] // 2), :]
        kmeans = self.get_clustering_model(top_half)
        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - int(non_player_cluster)
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame: np.ndarray, player_detections: dict[int, dict]) -> None:
        player_colors = [self.get_player_color(frame, p["bbox"]) for p in player_detections.values()]
        if len(player_colors) < 2:
            raise ValueError("Need at least two player detections to infer two team colors.")
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        self.kmeans.fit(player_colors)
        self.team_colors[1] = tuple(int(v) for v in self.kmeans.cluster_centers_[0])
        self.team_colors[2] = tuple(int(v) for v in self.kmeans.cluster_centers_[1])

    def get_player_team(self, frame: np.ndarray, player_bbox: list[float], player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        if self.kmeans is None:
            raise RuntimeError("Team colors are not assigned yet. Call assign_team_color first.")
        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        self.player_team_dict[player_id] = team_id
        return team_id
