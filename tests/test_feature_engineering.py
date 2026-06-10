from __future__ import annotations

import pandas as pd


def test_add_match_context_features(
    feature_engineering_module,
    sample_cups_df: pd.DataFrame,
    sample_matches_df: pd.DataFrame,
) -> None:
    result = feature_engineering_module.add_match_context_features(sample_cups_df, sample_matches_df)

    first = result.iloc[0]
    assert first["total_goals"] == 3
    assert first["goal_diff"] == 1
    assert first["home_win"] == 1
    assert first["is_draw"] == 0
    assert first["is_knockout_stage"] == 0
    assert first["tournament_matches_played"] == 64

    quarter_final = result.iloc[1]
    assert quarter_final["is_knockout_stage"] == 1


def test_add_team_form_features_has_prior_state_deltas(
    feature_engineering_module,
    sample_matches_df: pd.DataFrame,
) -> None:
    result = feature_engineering_module.add_team_form_features(sample_matches_df)

    first = result.iloc[0]
    assert first["home_prior_matches"] == 0
    assert first["away_prior_matches"] == 0
    assert first["home_prior_win_rate"] == 0.0
    assert first["away_prior_win_rate"] == 0.0

    second = result.iloc[1]
    assert second["home_prior_matches"] == 1
    assert second["away_prior_matches"] == 1
    assert second["away_prior_wins"] == 1
    assert second["prior_win_rate_delta"] < 0


def test_build_country_tournament_features_outputs_expected_flags(
    feature_engineering_module,
    sample_cups_df: pd.DataFrame,
    sample_matches_df: pd.DataFrame,
) -> None:
    result = feature_engineering_module.build_country_tournament_features(sample_cups_df, sample_matches_df)

    required_columns = {
        "Year",
        "Team",
        "matches_total",
        "wins_total",
        "home_win_rate",
        "away_win_rate",
        "home_away_win_rate_delta",
        "is_host_team",
        "is_world_cup_winner",
    }
    assert required_columns.issubset(set(result.columns))

    germany_2014 = result[(result["Year"] == "2014") & (result["Team"] == "Germany")].iloc[0]
    assert germany_2014["is_world_cup_winner"] == 1

    brazil_2014 = result[(result["Year"] == "2014") & (result["Team"] == "Brazil")]
    if not brazil_2014.empty:
        assert brazil_2014.iloc[0]["is_host_team"] == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
