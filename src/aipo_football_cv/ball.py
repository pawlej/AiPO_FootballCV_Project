from __future__ import annotations

from .geometry import get_center_bbox, measure_distance


class BallAssigner:
    """Assign ball possession to the nearest player foot within a distance threshold."""

    def __init__(self, max_distance: float = 60.0, min_frames_with_ball: int = 12):
        self.max_distance = max_distance
        self.min_frames_with_ball = min_frames_with_ball
        self.player_ball_possession_frames: dict[int, int] = {}

    def assign_ball_to_player(self, players: dict[int, dict], ball_bbox: list[float]) -> int:
        ball_position = get_center_bbox(ball_bbox)
        assigned_player = -1
        min_distance = float("inf")

        for player_id, player in players.items():
            bbox = player["bbox"]
            left_foot = (bbox[0], bbox[3])
            right_foot = (bbox[2], bbox[3])
            distance = min(measure_distance(left_foot, ball_position), measure_distance(right_foot, ball_position))

            if distance < self.max_distance and distance < min_distance:
                self.player_ball_possession_frames[player_id] = self.player_ball_possession_frames.get(player_id, 0) + 1
                if self.player_ball_possession_frames[player_id] >= self.min_frames_with_ball:
                    assigned_player = player_id
                    min_distance = distance

        for player_id in list(self.player_ball_possession_frames):
            if player_id not in players or (assigned_player != -1 and player_id != assigned_player):
                del self.player_ball_possession_frames[player_id]

        return assigned_player
