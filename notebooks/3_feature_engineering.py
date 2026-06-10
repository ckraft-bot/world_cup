from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class Paths:
    """Container for project paths used by this feature engineering workflow."""

    base_dir: Path
    data_dir: Path
    output_dir: Path


def resolve_paths() -> Paths:
    """Resolve repository-relative paths regardless of run location."""
    base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data" / "processed"
    output_dir = data_dir / "features"

    output_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        base_dir=base_dir,
        data_dir=data_dir,
        output_dir=output_dir,
    )

def load_processed_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed input tables for feature generation."""
    cups_df = pd.read_csv(paths.data_dir / "world_cups_clean.csv")
    matches_df = pd.read_csv(paths.data_dir / "world_cup_matches_clean.csv")
    players_df = pd.read_csv(paths.data_dir / "world_cup_players_clean.csv")

    matches_df["Datetime"] = pd.to_datetime(matches_df["Datetime"], errors="coerce")
    matches_df = matches_df.sort_values(["Datetime", "MatchID"]).reset_index(drop=True)

    logger.info("Loaded cups: {}", cups_df.shape)
    logger.info("Loaded matches: {}", matches_df.shape)
    logger.info("Loaded players: {}", players_df.shape)
    return cups_df, matches_df, players_df


def build_player_team_sheet_features(players_df: pd.DataFrame) -> pd.DataFrame:
    """Build match-team features from lineup and event signals."""
    players = players_df.copy()
    players["Event"] = players["Event"].fillna("")

    players["is_starter"] = players["LineUp"].eq("S").astype(int)
    players["is_substitute"] = players["LineUp"].eq("N").astype(int)
    players["is_goalkeeper"] = players["PlayerPosition"].eq("GK").astype(int)
    players["is_captain"] = players["PlayerPosition"].eq("C").astype(int)

    players["event_goals"] = players["Event"].str.count(r"G\d+")
    players["event_yellow_cards"] = players["Event"].str.count(r"Y")
    players["event_red_cards"] = players["Event"].str.count(r"R|SY")
    players["event_penalties"] = players["Event"].str.count(r"P")

    team_sheet = (
        players.groupby(["MatchID", "TeamInitials"], as_index=False)
        .agg(
            starters=("is_starter", "sum"),
            substitutes=("is_substitute", "sum"),
            goalkeepers=("is_goalkeeper", "sum"),
            captains=("is_captain", "sum"),
            bench_size=("PlayerName", "count"),
            event_goal_tokens=("event_goals", "sum"),
            event_yellow_cards=("event_yellow_cards", "sum"),
            event_red_cards=("event_red_cards", "sum"),
            event_penalties=("event_penalties", "sum"),
        )
        .sort_values(["MatchID", "TeamInitials"])
    )

    return team_sheet


def merge_team_sheet_features(matches_df: pd.DataFrame, team_sheet: pd.DataFrame) -> pd.DataFrame:
    """Merge per-match team sheet features onto home and away sides."""
    match_features = matches_df.copy()

    home_sheet = team_sheet.add_prefix("home_").rename(
        columns={"home_MatchID": "MatchID", "home_TeamInitials": "HomeTeamInitials"}
    )
    away_sheet = team_sheet.add_prefix("away_").rename(
        columns={"away_MatchID": "MatchID", "away_TeamInitials": "AwayTeamInitials"}
    )

    match_features = match_features.merge(home_sheet, on=["MatchID", "HomeTeamInitials"], how="left")
    match_features = match_features.merge(away_sheet, on=["MatchID", "AwayTeamInitials"], how="left")

    return match_features


def add_match_context_features(cups_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Create base match context features and join tournament-level metadata."""
    features = matches_df.copy()

    features["total_goals"] = features["HomeTeamGoals"] + features["AwayTeamGoals"]
    features["goal_diff"] = features["HomeTeamGoals"] - features["AwayTeamGoals"]
    features["is_draw"] = features["goal_diff"].eq(0).astype(int)
    features["home_win"] = features["goal_diff"].gt(0).astype(int)
    features["away_win"] = features["goal_diff"].lt(0).astype(int)

    features["match_month"] = features["Datetime"].dt.month
    features["match_day_of_week"] = features["Datetime"].dt.dayofweek
    features["is_weekend_match"] = features["match_day_of_week"].isin([5, 6]).astype(int)

    features["attendance_log1p"] = np.log1p(features["Attendance"].fillna(0))
    features["half_time_goal_diff"] = (
        features["HalftimeHomeGoals"] - features["HalftimeAwayGoals"]
    )

    knockout_keywords = ["Final", "Semi", "Quarter", "third place", "Round of", "Preliminary"]
    features["is_knockout_stage"] = (
        features["Stage"].fillna("").str.contains("|".join(knockout_keywords), case=False, regex=True)
    ).astype(int)

    cups_meta = cups_df[["Year", "QualifiedTeams", "MatchesPlayed", "GoalsScored", "Attendance"]].rename(
        columns={
            "MatchesPlayed": "tournament_matches_played",
            "GoalsScored": "tournament_goals_scored",
            "Attendance": "tournament_attendance",
        }
    )
    features = features.merge(cups_meta, on="Year", how="left")

    return features


def add_team_form_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling pre-match team form features for home and away teams."""
    features = matches_df.copy()

    team_state: dict[str, dict[str, float]] = {}
    home_side_state: dict[str, dict[str, float]] = {}
    away_side_state: dict[str, dict[str, float]] = {}
    team_win_attendance_state: dict[str, dict[str, float]] = {}

    home_prior_matches: list[int] = []
    home_prior_wins: list[int] = []
    home_prior_goals_for: list[int] = []
    home_prior_goals_against: list[int] = []

    away_prior_matches: list[int] = []
    away_prior_wins: list[int] = []
    away_prior_goals_for: list[int] = []
    away_prior_goals_against: list[int] = []

    home_team_prior_home_matches: list[int] = []
    home_team_prior_home_wins: list[int] = []
    away_team_prior_away_matches: list[int] = []
    away_team_prior_away_wins: list[int] = []

    home_team_prior_win_attendance_avg: list[float] = []
    away_team_prior_win_attendance_avg: list[float] = []

    def init_team(team_name: str) -> None:
        if team_name not in team_state:
            team_state[team_name] = {"matches": 0, "wins": 0, "goals_for": 0, "goals_against": 0}
        if team_name not in home_side_state:
            home_side_state[team_name] = {"matches": 0, "wins": 0}
        if team_name not in away_side_state:
            away_side_state[team_name] = {"matches": 0, "wins": 0}
        if team_name not in team_win_attendance_state:
            team_win_attendance_state[team_name] = {"wins": 0, "attendance_sum": 0.0}

    for row in features.itertuples(index=False):
        home_team = row.HomeTeamName
        away_team = row.AwayTeamName

        init_team(home_team)
        init_team(away_team)

        home_snapshot = team_state[home_team]
        away_snapshot = team_state[away_team]
        home_side_snapshot = home_side_state[home_team]
        away_side_snapshot = away_side_state[away_team]
        home_win_attendance_snapshot = team_win_attendance_state[home_team]
        away_win_attendance_snapshot = team_win_attendance_state[away_team]

        home_prior_matches.append(int(home_snapshot["matches"]))
        home_prior_wins.append(int(home_snapshot["wins"]))
        home_prior_goals_for.append(int(home_snapshot["goals_for"]))
        home_prior_goals_against.append(int(home_snapshot["goals_against"]))

        away_prior_matches.append(int(away_snapshot["matches"]))
        away_prior_wins.append(int(away_snapshot["wins"]))
        away_prior_goals_for.append(int(away_snapshot["goals_for"]))
        away_prior_goals_against.append(int(away_snapshot["goals_against"]))

        home_team_prior_home_matches.append(int(home_side_snapshot["matches"]))
        home_team_prior_home_wins.append(int(home_side_snapshot["wins"]))
        away_team_prior_away_matches.append(int(away_side_snapshot["matches"]))
        away_team_prior_away_wins.append(int(away_side_snapshot["wins"]))

        if home_win_attendance_snapshot["wins"] > 0:
            home_team_prior_win_attendance_avg.append(
                float(home_win_attendance_snapshot["attendance_sum"] / home_win_attendance_snapshot["wins"])
            )
        else:
            home_team_prior_win_attendance_avg.append(0.0)

        if away_win_attendance_snapshot["wins"] > 0:
            away_team_prior_win_attendance_avg.append(
                float(away_win_attendance_snapshot["attendance_sum"] / away_win_attendance_snapshot["wins"])
            )
        else:
            away_team_prior_win_attendance_avg.append(0.0)

        home_goals_raw = pd.to_numeric(row.HomeTeamGoals, errors="coerce")
        away_goals_raw = pd.to_numeric(row.AwayTeamGoals, errors="coerce")
        home_goals = 0 if pd.isna(home_goals_raw) else int(home_goals_raw)
        away_goals = 0 if pd.isna(away_goals_raw) else int(away_goals_raw)
        attendance_raw = pd.to_numeric(row.Attendance, errors="coerce")
        attendance = 0.0 if pd.isna(attendance_raw) else float(attendance_raw)

        team_state[home_team]["matches"] += 1
        team_state[away_team]["matches"] += 1
        home_side_state[home_team]["matches"] += 1
        away_side_state[away_team]["matches"] += 1

        team_state[home_team]["goals_for"] += home_goals
        team_state[home_team]["goals_against"] += away_goals

        team_state[away_team]["goals_for"] += away_goals
        team_state[away_team]["goals_against"] += home_goals

        if home_goals > away_goals:
            team_state[home_team]["wins"] += 1
            home_side_state[home_team]["wins"] += 1
            team_win_attendance_state[home_team]["wins"] += 1
            team_win_attendance_state[home_team]["attendance_sum"] += attendance
        elif away_goals > home_goals:
            team_state[away_team]["wins"] += 1
            away_side_state[away_team]["wins"] += 1
            team_win_attendance_state[away_team]["wins"] += 1
            team_win_attendance_state[away_team]["attendance_sum"] += attendance

    features["home_prior_matches"] = home_prior_matches
    features["home_prior_wins"] = home_prior_wins
    features["home_prior_goals_for"] = home_prior_goals_for
    features["home_prior_goals_against"] = home_prior_goals_against

    features["away_prior_matches"] = away_prior_matches
    features["away_prior_wins"] = away_prior_wins
    features["away_prior_goals_for"] = away_prior_goals_for
    features["away_prior_goals_against"] = away_prior_goals_against

    features["home_team_prior_home_matches"] = home_team_prior_home_matches
    features["home_team_prior_home_wins"] = home_team_prior_home_wins
    features["away_team_prior_away_matches"] = away_team_prior_away_matches
    features["away_team_prior_away_wins"] = away_team_prior_away_wins

    features["home_team_prior_win_attendance_avg"] = home_team_prior_win_attendance_avg
    features["away_team_prior_win_attendance_avg"] = away_team_prior_win_attendance_avg

    home_denom = features["home_prior_matches"].replace(0, np.nan)
    away_denom = features["away_prior_matches"].replace(0, np.nan)

    features["home_prior_win_rate"] = (features["home_prior_wins"] / home_denom).fillna(0.0)
    features["away_prior_win_rate"] = (features["away_prior_wins"] / away_denom).fillna(0.0)

    home_side_denom = features["home_team_prior_home_matches"].replace(0, np.nan)
    away_side_denom = features["away_team_prior_away_matches"].replace(0, np.nan)

    features["home_team_prior_home_win_rate"] = (
        features["home_team_prior_home_wins"] / home_side_denom
    ).fillna(0.0)
    features["away_team_prior_away_win_rate"] = (
        features["away_team_prior_away_wins"] / away_side_denom
    ).fillna(0.0)

    features["home_prior_goal_balance"] = (
        features["home_prior_goals_for"] - features["home_prior_goals_against"]
    )
    features["away_prior_goal_balance"] = (
        features["away_prior_goals_for"] - features["away_prior_goals_against"]
    )

    features["prior_win_rate_delta"] = (
        features["home_prior_win_rate"] - features["away_prior_win_rate"]
    )
    features["prior_side_win_rate_delta"] = (
        features["home_team_prior_home_win_rate"] - features["away_team_prior_away_win_rate"]
    )
    features["prior_goal_balance_delta"] = (
        features["home_prior_goal_balance"] - features["away_prior_goal_balance"]
    )
    features["prior_win_attendance_avg_delta"] = (
        features["home_team_prior_win_attendance_avg"] - features["away_team_prior_win_attendance_avg"]
    )

    return features


def build_team_year_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build team-year aggregate performance features from match outcomes."""
    home_view = matches_df[["Year", "HomeTeamName", "HomeTeamGoals", "AwayTeamGoals"]].copy()
    home_view.columns = ["Year", "Team", "GoalsFor", "GoalsAgainst"]

    away_view = matches_df[["Year", "AwayTeamName", "AwayTeamGoals", "HomeTeamGoals"]].copy()
    away_view.columns = ["Year", "Team", "GoalsFor", "GoalsAgainst"]

    stacked = pd.concat([home_view, away_view], ignore_index=True)
    stacked["Win"] = (stacked["GoalsFor"] > stacked["GoalsAgainst"]).astype(int)
    stacked["Draw"] = (stacked["GoalsFor"] == stacked["GoalsAgainst"]).astype(int)
    stacked["Loss"] = (stacked["GoalsFor"] < stacked["GoalsAgainst"]).astype(int)

    team_year = (
        stacked.groupby(["Year", "Team"], as_index=False)
        .agg(
            matches=("Team", "count"),
            wins=("Win", "sum"),
            draws=("Draw", "sum"),
            losses=("Loss", "sum"),
            goals_for=("GoalsFor", "sum"),
            goals_against=("GoalsAgainst", "sum"),
        )
        .sort_values(["Year", "wins", "goals_for"], ascending=[True, False, False])
    )

    team_year["goal_balance"] = team_year["goals_for"] - team_year["goals_against"]
    team_year["points"] = team_year["wins"] * 3 + team_year["draws"]
    team_year["points_per_match"] = team_year["points"] / team_year["matches"]

    return team_year


def build_country_tournament_features(cups_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build country-year features for predicting World Cup tournament winners."""
    home_view = matches_df[
        ["Year", "HomeTeamName", "HomeTeamGoals", "AwayTeamGoals", "Attendance"]
    ].copy()
    home_view = home_view.rename(
        columns={
            "HomeTeamName": "Team",
            "HomeTeamGoals": "GoalsFor",
            "AwayTeamGoals": "GoalsAgainst",
        }
    )
    home_view["side"] = "home"

    away_view = matches_df[
        ["Year", "AwayTeamName", "AwayTeamGoals", "HomeTeamGoals", "Attendance"]
    ].copy()
    away_view = away_view.rename(
        columns={
            "AwayTeamName": "Team",
            "AwayTeamGoals": "GoalsFor",
            "HomeTeamGoals": "GoalsAgainst",
        }
    )
    away_view["side"] = "away"

    team_match = pd.concat([home_view, away_view], ignore_index=True)
    team_match["Win"] = (team_match["GoalsFor"] > team_match["GoalsAgainst"]).astype(int)
    team_match["Draw"] = (team_match["GoalsFor"] == team_match["GoalsAgainst"]).astype(int)
    team_match["Loss"] = (team_match["GoalsFor"] < team_match["GoalsAgainst"]).astype(int)

    summary = (
        team_match.groupby(["Year", "Team"], as_index=False)
        .agg(
            matches_total=("Team", "count"),
            wins_total=("Win", "sum"),
            draws_total=("Draw", "sum"),
            losses_total=("Loss", "sum"),
            goals_for_total=("GoalsFor", "sum"),
            goals_against_total=("GoalsAgainst", "sum"),
            avg_attendance=("Attendance", "mean"),
        )
    )

    side_summary = (
        team_match.groupby(["Year", "Team", "side"], as_index=False)
        .agg(
            matches=("Team", "count"),
            wins=("Win", "sum"),
            avg_attendance=("Attendance", "mean"),
        )
    )

    side_wide = (
        side_summary.pivot(index=["Year", "Team"], columns="side", values=["matches", "wins", "avg_attendance"])
        .fillna(0)
    )
    side_wide.columns = [f"{side}_{metric}" for metric, side in side_wide.columns]
    side_wide = side_wide.reset_index()

    win_attendance = (
        team_match[team_match["Win"].eq(1)]
        .groupby(["Year", "Team"], as_index=False)
        .agg(win_attendance_avg=("Attendance", "mean"))
    )

    side_win_attendance = (
        team_match[team_match["Win"].eq(1)]
        .groupby(["Year", "Team", "side"], as_index=False)
        .agg(win_attendance_avg=("Attendance", "mean"))
    )
    side_win_wide = (
        side_win_attendance.pivot(index=["Year", "Team"], columns="side", values="win_attendance_avg")
        .rename(columns={"home": "home_win_attendance_avg", "away": "away_win_attendance_avg"})
        .fillna(0)
        .reset_index()
    )

    features = summary.merge(side_wide, on=["Year", "Team"], how="left")
    features = features.merge(win_attendance, on=["Year", "Team"], how="left")
    features = features.merge(side_win_wide, on=["Year", "Team"], how="left")
    features = features.fillna(0)

    features["goal_balance_total"] = features["goals_for_total"] - features["goals_against_total"]
    features["win_rate_total"] = features["wins_total"] / features["matches_total"].replace(0, np.nan)
    features["home_win_rate"] = features["home_wins"] / features["home_matches"].replace(0, np.nan)
    features["away_win_rate"] = features["away_wins"] / features["away_matches"].replace(0, np.nan)
    features["home_away_win_rate_delta"] = features["home_win_rate"] - features["away_win_rate"]
    features["attendance_win_minus_avg"] = features["win_attendance_avg"] - features["avg_attendance"]
    features["home_away_win_attendance_delta"] = (
        features["home_win_attendance_avg"] - features["away_win_attendance_avg"]
    )

    cups_meta = cups_df[["Year", "Country", "Winner", "QualifiedTeams", "MatchesPlayed", "Attendance"]].copy()
    cups_meta = cups_meta.rename(
        columns={
            "Country": "HostCountry",
            "Attendance": "tournament_attendance",
            "MatchesPlayed": "tournament_matches_played",
        }
    )
    features = features.merge(cups_meta, on="Year", how="left")

    features["is_host_team"] = features["Team"].eq(features["HostCountry"]).astype(int)
    features["is_world_cup_winner"] = features["Team"].eq(features["Winner"]).astype(int)

    for col in [
        "win_rate_total",
        "home_win_rate",
        "away_win_rate",
        "home_away_win_rate_delta",
        "attendance_win_minus_avg",
        "home_away_win_attendance_delta",
    ]:
        features[col] = features[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features.sort_values(["Year", "Team"]).reset_index(drop=True)


def save_outputs(
    paths: Paths,
    match_features: pd.DataFrame,
    team_year: pd.DataFrame,
    country_tournament: pd.DataFrame,
) -> None:
    """Persist generated feature sets to disk."""
    match_out = paths.output_dir / "match_features.csv"
    team_year_out = paths.output_dir / "team_year_features.csv"
    country_tournament_out = paths.output_dir / "country_tournament_features.csv"

    match_features.to_csv(match_out, index=False)
    team_year.to_csv(team_year_out, index=False)
    country_tournament.to_csv(country_tournament_out, index=False)

    logger.info("Saved match features: {}", match_out)
    logger.info("Saved team-year features: {}", team_year_out)
    logger.info("Saved country-tournament features: {}", country_tournament_out)


def main() -> None:
    """Entry point for feature engineering workflow."""
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    paths = resolve_paths()
    cups_df, matches_df, players_df = load_processed_data(paths)

    team_sheet = build_player_team_sheet_features(players_df)

    match_features = add_match_context_features(cups_df, matches_df)
    match_features = add_team_form_features(match_features)
    match_features = merge_team_sheet_features(match_features, team_sheet)

    team_year_features = build_team_year_features(matches_df)
    country_tournament_features = build_country_tournament_features(cups_df, matches_df)

    save_outputs(paths, match_features, team_year_features, country_tournament_features)

    logger.info("Feature engineering complete.")
    logger.info("match_features columns: {}", len(match_features.columns))
    logger.info("team_year_features columns: {}", len(team_year_features.columns))
    logger.info("country_tournament_features columns: {}", len(country_tournament_features.columns))


if __name__ == "__main__":
    print(f"__file__ = {__file__}")

    paths = resolve_paths()
    print(f"base_dir = {paths.base_dir}")
    print(f"data_dir = {paths.data_dir}")

    main()
