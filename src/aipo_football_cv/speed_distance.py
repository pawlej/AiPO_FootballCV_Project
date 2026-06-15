from __future__ import annotations

import cv2

from .geometry import get_foot_position, measure_distance
from .tracker import Tracks


class SpeedAndDistanceEstimator:
    """Estimate player speed and distance from transformed pitch positions.

    If homography positions are unavailable, pixel positions are used as a fallback.
    Real metric interpretation is better when ``position_transformed`` is present.
    """

    def __init__(self, frame_rate: float = 24.0, meter_scale: float = 0.01):
        self.frame_rate = frame_rate
        self.meter_scale = meter_scale
        self.previous_positions: dict[int, list[float] | tuple[float, float]] = {}
        self.player_stats: dict[int, dict[str, float]] = {}

    def add_speed_and_distance_to_tracks(self, tracks: Tracks) -> None:
        total_distance: dict[int, float] = {}
        for frame_tracks in tracks.get("players", []):
            for track_id, track_info in frame_tracks.items():
                current_position = track_info.get("position_transformed") or track_info.get("position")
                if current_position is None:
                    continue
                if track_id in self.previous_positions:
                    previous = self.previous_positions[track_id]
                    distance = measure_distance(previous, current_position) * self.meter_scale
                    total_distance[track_id] = total_distance.get(track_id, 0.0) + distance
                    speed_kmh = distance * self.frame_rate * 3.6
                    track_info["speed"] = speed_kmh
                    track_info["distance"] = total_distance[track_id]

                    stats = self.player_stats.setdefault(track_id, {"total_distance": 0.0, "max_speed": 0.0, "speed_sum": 0.0, "frame_count": 0.0})
                    stats["total_distance"] = total_distance[track_id]
                    stats["max_speed"] = max(stats["max_speed"], speed_kmh)
                    stats["speed_sum"] += speed_kmh
                    stats["frame_count"] += 1
                self.previous_positions[track_id] = current_position

    @staticmethod
    def draw_speed_and_distance(frames: list, tracks: Tracks) -> list:
        for frame_num, frame in enumerate(frames):
            for track_info in tracks.get("players", [])[frame_num].values():
                if "speed" not in track_info:
                    continue
                bbox = track_info["bbox"]
                x, y = get_foot_position(bbox)
                y += 45
                cv2.putText(frame, f"{track_info['speed']:.1f} km/h", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
                cv2.putText(frame, f"{track_info['distance']:.1f} m", (x, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
        return frames

    def get_player_stats(self) -> dict[int, dict[str, float]]:
        return {
            player_id: {
                "total_distance": stats["total_distance"],
                "max_speed": stats["max_speed"],
                "avg_speed": stats["speed_sum"] / stats["frame_count"] if stats["frame_count"] else 0.0,
            }
            for player_id, stats in self.player_stats.items()
        }
