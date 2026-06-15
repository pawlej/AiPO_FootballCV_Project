from __future__ import annotations

import cv2

from .tracker import Tracks


class PassesAndPossessionCounter:
    """Count approximate passes, losses and ball possession based on assigned ball owner."""

    def __init__(self, fps: float = 24.0, pass_threshold_frames: int = 25):
        self.fps = fps
        self.team1_passes: list[tuple[int, int]] = []
        self.team2_passes: list[tuple[int, int]] = []
        self.team1_losses: list[tuple[int, int]] = []
        self.team2_losses: list[tuple[int, int]] = []
        self.last_pass_frame = -1
        self.pass_threshold = pass_threshold_frames
        self.ball_possession: list[int | None] = []
        self.player_possession_time: dict[int, int] = {}
        self.player_passes: dict[int, int] = {}

    def detect_passes(self, track_id: int, player: dict, frame_num: int, previous_ball_owner: tuple[int, int] | None = None) -> tuple[int, int]:
        current_ball_owner = (track_id, player["team"])
        if previous_ball_owner is None:
            return current_ball_owner
        if previous_ball_owner != current_ball_owner and frame_num - self.last_pass_frame > self.pass_threshold:
            if previous_ball_owner[1] == current_ball_owner[1]:
                if previous_ball_owner[1] == 1:
                    self.team1_passes.append(previous_ball_owner)
                else:
                    self.team2_passes.append(previous_ball_owner)
                self.player_passes[previous_ball_owner[0]] = self.player_passes.get(previous_ball_owner[0], 0) + 1
            else:
                if previous_ball_owner[1] == 1:
                    self.team1_losses.append(previous_ball_owner)
                else:
                    self.team2_losses.append(previous_ball_owner)
            self.last_pass_frame = frame_num
            return current_ball_owner
        return previous_ball_owner

    def draw_passes_and_possession(self, tracks: Tracks, frames: list) -> list:
        previous_ball_owner: tuple[int, int] | None = None
        team_ball_owner: int | None = None
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 200), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            for track_id, player in tracks["players"][frame_num].items():
                if player.get("has_ball", False):
                    self.player_possession_time[track_id] = self.player_possession_time.get(track_id, 0) + 1
                    previous_ball_owner = self.detect_passes(track_id, player, frame_num, previous_ball_owner)
                    team_ball_owner = player["team"]

            self.ball_possession.append(team_ball_owner)
            team1_frames = self.ball_possession.count(1)
            team2_frames = self.ball_possession.count(2)
            total = team1_frames + team2_frames + 1e-9
            team1_percent = team1_frames / total * 100
            team2_percent = team2_frames / total * 100

            cv2.putText(frame, f"Team 1 passes: {len(self.team1_passes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Team 2 passes: {len(self.team2_passes)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, "Ball possession:", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            bar_x, bar_y, bar_width, bar_height = 10, 145, 400, 30
            team1_color = (0, 0, 255)
            team2_color = (255, 0, 0)
            team1_bar_width = int(bar_width * team1_percent / 100)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + team1_bar_width, bar_y + bar_height), team1_color, -1)
            cv2.rectangle(frame, (bar_x + team1_bar_width, bar_y), (bar_x + bar_width, bar_y + bar_height), team2_color, -1)
            cv2.putText(frame, f"Team 1: {team1_percent:.1f}%", (bar_x + 5, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, team1_color, 2)
            cv2.putText(frame, f"Team 2: {team2_percent:.1f}%", (bar_x + bar_width - 150, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, team2_color, 2)
            output_frames.append(frame)

        return output_frames

    def get_player_stats(self) -> dict[int, dict[str, float]]:
        return {
            player_id: {
                "possession_time": possession_frames / self.fps,
                "passes": self.player_passes.get(player_id, 0),
            }
            for player_id, possession_frames in self.player_possession_time.items()
        }
