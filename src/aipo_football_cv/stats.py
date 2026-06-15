from __future__ import annotations

import pandas as pd

from .possession import PassesAndPossessionCounter


def build_stats_tables(
    possession_counter: PassesAndPossessionCounter,
    player_team_dict: dict[int, int],
    player_speed_distance_stats: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team1_possession = round(possession_counter.ball_possession.count(1) / len(possession_counter.ball_possession) * 100, 2) if possession_counter.ball_possession else 0
    team2_possession = round(possession_counter.ball_possession.count(2) / len(possession_counter.ball_possession) * 100, 2) if possession_counter.ball_possession else 0

    team_rows = []
    for team_id, possession_percent, passes in [
        (1, team1_possession, len(possession_counter.team1_passes)),
        (2, team2_possession, len(possession_counter.team2_passes)),
    ]:
        players = [player_id for player_id, team in player_team_dict.items() if team == team_id]
        avg_speed = sum(player_speed_distance_stats.get(pid, {}).get("avg_speed", 0.0) for pid in players) / len(players) if players else 0.0
        total_distance = sum(player_speed_distance_stats.get(pid, {}).get("total_distance", 0.0) for pid in players)
        team_rows.append(
            {
                "team": team_id,
                "ball_possession_percent": possession_percent,
                "passes": passes,
                "avg_speed_kmh": round(avg_speed, 2),
                "total_distance_m": round(total_distance, 2),
            }
        )

    player_rows = []
    player_possession_stats = possession_counter.get_player_stats()
    for player_id, stats in player_possession_stats.items():
        losses = possession_counter.team1_losses.count((player_id, 1)) + possession_counter.team2_losses.count((player_id, 2))
        speed_stats = player_speed_distance_stats.get(player_id, {})
        player_rows.append(
            {
                "player_id": player_id,
                "team": player_team_dict.get(player_id, "N/A"),
                "possession_time_s": round(stats["possession_time"], 2),
                "passes": stats["passes"],
                "losses": losses,
                "max_speed_kmh": round(speed_stats.get("max_speed", 0.0), 2),
                "avg_speed_kmh": round(speed_stats.get("avg_speed", 0.0), 2),
                "distance_m": round(speed_stats.get("total_distance", 0.0), 2),
            }
        )

    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)
