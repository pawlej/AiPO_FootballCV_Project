from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ball import BallAssigner
from .possession import PassesAndPossessionCounter
from .speed_distance import SpeedAndDistanceEstimator
from .stats import build_stats_tables
from .team import TeamAssigner
from .tracker import Tracker
from .video import get_video_fps, load_video_frames, save_video
from .view_transformer import ViewTransformer


@dataclass
class PipelineResult:
    output_video_path: Path
    team_stats_path: Path
    player_stats_path: Path
    tracks_path: Path | None


def run_pipeline(
    video_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path = "outputs",
    confidence: float = 0.3,
    batch_size: int = 20,
    max_frames: int | None = None,
    stride: int = 1,
    use_stub: bool = False,
    stub_path: str | Path | None = "stubs/tracks.pkl",
    enable_view_transform: bool = False,
    roboflow_api_key: str | None = None,
    meter_scale: float = 0.01,
) -> PipelineResult:
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_video_fps(video_path)
    frames = load_video_frames(video_path, max_frames=max_frames, stride=stride)
    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")

    tracker = Tracker(model_path=model_path, confidence=confidence, batch_size=batch_size)
    tracks = tracker.get_object_tracks(frames, read_from_stub=use_stub, stub_path=stub_path)
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.ball_pos_interpolation(tracks["ball"])

    if enable_view_transform:
        transformer = ViewTransformer(roboflow_api_key=roboflow_api_key)
        transformer.add_transformed_position_to_tracks(tracks, frames)

    speed_estimator = SpeedAndDistanceEstimator(frame_rate=fps / stride, meter_scale=meter_scale)
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    first_frame_with_players = next((i for i, item in enumerate(tracks["players"]) if len(item) >= 2), None)
    if first_frame_with_players is None:
        raise ValueError("Could not find a frame with at least two players for team assignment.")
    team_assigner.assign_team_color(frames[first_frame_with_players], tracks["players"][first_frame_with_players])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track["bbox"], player_id)
            track["team"] = team
            track["team_color"] = team_assigner.team_colors[team]

    ball_assigner = BallAssigner()
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_items = tracks["ball"][frame_num]
        if not ball_items:
            continue
        ball_bbox = ball_items[1]["bbox"]
        player_id = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        if player_id != -1 and player_id in tracks["players"][frame_num]:
            tracks["players"][frame_num][player_id]["has_ball"] = True

    possession_counter = PassesAndPossessionCounter(fps=fps / stride)
    frames_with_overlay = possession_counter.draw_passes_and_possession(tracks, frames)
    annotated_frames = tracker.draw_annotations(frames_with_overlay, tracks)
    speed_estimator.draw_speed_and_distance(annotated_frames, tracks)

    output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
    save_video(annotated_frames, output_video_path, fps=fps / stride)

    team_stats, player_stats = build_stats_tables(
        possession_counter,
        team_assigner.player_team_dict,
        speed_estimator.get_player_stats(),
    )
    team_stats_path = output_dir / f"{video_path.stem}_team_stats.csv"
    player_stats_path = output_dir / f"{video_path.stem}_player_stats.csv"
    team_stats.to_csv(team_stats_path, index=False)
    player_stats.to_csv(player_stats_path, index=False)

    return PipelineResult(output_video_path, team_stats_path, player_stats_path, Path(stub_path) if stub_path else None)
