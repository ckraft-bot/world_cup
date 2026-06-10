# Feature Engineering Write-Up

## Objective
This feature engineering pipeline transforms cleaned World Cup tournament, match, and player data into model-ready datasets for both match-level and team-year analysis.

It produces two feature outputs:
- `data/processed/features/match_features.csv`
- `data/processed/features/team_year_features.csv`

## Pipeline Inputs
The workflow loads:
- `data/processed/world_cups_clean.csv`
- `data/processed/world_cup_matches_clean.csv`
- `data/processed/world_cup_players_clean.csv`

## Feature Families and Rationale

### 1. Match Context Features
Built in `add_match_context_features`.

Key fields:
- `total_goals`, `goal_diff`, `is_draw`, `home_win`, `away_win`
- `match_month`, `match_day_of_week`, `is_weekend_match`
- `attendance_log1p`
- `half_time_goal_diff`
- `is_knockout_stage`
- Tournament metadata joins:
  - `QualifiedTeams`
  - `tournament_matches_played`
  - `tournament_goals_scored`
  - `tournament_attendance`

Why these were selected:
- Capture match intensity and timing context.
- Encode structural differences between group and knockout football.
- Stabilize heavy-tailed attendance values using log transform.
- Add macro tournament context per match.

### 2. Rolling Team Form Features
Built in `add_team_form_features`.

Key fields:
- Prior home/away state before each match:
  - `home_prior_matches`, `away_prior_matches`
  - `home_prior_wins`, `away_prior_wins`
  - `home_prior_goals_for`, `away_prior_goals_for`
  - `home_prior_goals_against`, `away_prior_goals_against`
- Rate and balance fields:
  - `home_prior_win_rate`, `away_prior_win_rate`
  - `home_prior_goal_balance`, `away_prior_goal_balance`
- Relative deltas:
  - `prior_win_rate_delta`
  - `prior_goal_balance_delta`

Additional side-specific winning features:
- `home_team_prior_home_matches`, `home_team_prior_home_wins`
- `away_team_prior_away_matches`, `away_team_prior_away_wins`
- `home_team_prior_home_win_rate`, `away_team_prior_away_win_rate`
- `prior_side_win_rate_delta`

Attendance-on-winning features:
- `home_team_prior_win_attendance_avg`
- `away_team_prior_win_attendance_avg`
- `prior_win_attendance_avg_delta`

Why these were selected:
- Represent pre-match momentum and form.
- Encode directional strength differences between teams.
- Provide interpretable relative signals for modeling.
- Add explicit home/away winning tendencies per team.
- Add crowd-size context linked to each team's historical wins.

### 3. Team Sheet and Event Features
Built in `build_player_team_sheet_features`, then merged via `merge_team_sheet_features`.

Key fields per team per match:
- Squad structure:
  - `starters`, `substitutes`, `goalkeepers`, `captains`, `bench_size`
- Event aggregates:
  - `event_goal_tokens`
  - `event_yellow_cards`
  - `event_red_cards`
  - `event_penalties`

Merged as prefixed sides:
- `home_*`
- `away_*`

Why these were selected:
- Convert row-level player/event logs into compact team-level indicators.
- Preserve side-specific effects.
- Support tactical and discipline signal extraction.

### 4. Team-Year Aggregate Features
Built in `build_team_year_features`.

Key fields:
- `matches`, `wins`, `draws`, `losses`
- `goals_for`, `goals_against`, `goal_balance`
- `points`, `points_per_match`

Why these were selected:
- Build season-style team performance profiles by year.
- Enable trend analysis and historical strength benchmarking.

## Interpretation Guide

- `prior_win_rate_delta > 0`:
  Home team entered with stronger recent historical form than away team.

- `prior_side_win_rate_delta > 0`:
  Home side has historically been stronger in home fixtures than the away side has been in away fixtures.

- `prior_goal_balance_delta > 0`:
  Home team had a stronger prior goal margin profile.

- `prior_win_attendance_avg_delta > 0`:
  Home side historically wins in higher-attendance environments than the away side.

- `is_knockout_stage = 1`:
  Match belongs to a high-stakes phase that often changes tactical behavior.

- `attendance_log1p`:
  Normalized attendance signal; higher means bigger crowds with reduced skew impact.

- `home_event_red_cards` / `away_event_red_cards`:
  Discipline risk markers that can relate to match control and outcome volatility.

- `points_per_match`:
  Compact yearly team-performance metric for ranking and comparison.

## Selection Notes
Current feature engineering is broad and suitable for descriptive analytics plus exploratory modeling.

For strict pre-match prediction tasks, separate leakage-prone post-match fields from pre-match inputs.

Potential leakage fields for pre-match outcome prediction include:
- `HomeTeamGoals`, `AwayTeamGoals`
- `total_goals`, `goal_diff`
- `home_win`, `away_win`, `is_draw`
- `half_time_goal_diff`
- Event-derived fields if they encode in-match outcomes

Current model-focused selection includes attendance signals that are intended to be known or estimated pre-match:
- `attendance_log1p`
- historical win-attendance averages and delta fields listed above

## Output Governance
Outputs are saved in `data/processed/features/` with stable file names to support reproducibility and downstream model governance:
- `match_features.csv`
- `team_year_features.csv`
