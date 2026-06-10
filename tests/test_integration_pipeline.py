from __future__ import annotations

import pandas as pd


def test_feature_engineering_smoke_pipeline(
    feature_engineering_module,
    sample_cups_df: pd.DataFrame,
    sample_matches_df: pd.DataFrame,
) -> None:
    players_df = pd.DataFrame(
        [
            {"MatchID": 1, "TeamInitials": "BRA", "LineUp": "S", "PlayerPosition": "GK", "PlayerName": "P1", "Event": ""},
            {"MatchID": 1, "TeamInitials": "ESP", "LineUp": "S", "PlayerPosition": "C", "PlayerName": "P2", "Event": "G45 Y"},
            {"MatchID": 2, "TeamInitials": "ESP", "LineUp": "N", "PlayerPosition": "MF", "PlayerName": "P3", "Event": "R"},
            {"MatchID": 2, "TeamInitials": "BRA", "LineUp": "S", "PlayerPosition": "GK", "PlayerName": "P4", "Event": "P"},
            {"MatchID": 3, "TeamInitials": "GER", "LineUp": "S", "PlayerPosition": "GK", "PlayerName": "P5", "Event": "G90"},
            {"MatchID": 3, "TeamInitials": "ARG", "LineUp": "S", "PlayerPosition": "DF", "PlayerName": "P6", "Event": ""},
        ]
    )

    team_sheet = feature_engineering_module.build_player_team_sheet_features(players_df)
    match_features = feature_engineering_module.add_match_context_features(sample_cups_df, sample_matches_df)
    match_features = feature_engineering_module.add_team_form_features(match_features)
    merged_features = feature_engineering_module.merge_team_sheet_features(match_features, team_sheet)

    assert len(merged_features) == len(sample_matches_df)
    assert "home_starters" in merged_features.columns
    assert "away_event_goal_tokens" in merged_features.columns
    assert "prior_win_rate_delta" in merged_features.columns


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
