# Model Training and Evaluation Write-Up

## Objective
This stage trains and evaluates predictive models using engineered World Cup match features, with emphasis on reproducibility, leakage control, and model governance.

Primary training script:
- `notebooks/4_model_training_and_evaluation.py`

Primary inputs:
- `data/processed/features/match_features.csv`

Primary target:
- `home_win` (binary classification)

## Modeling Scope
The current modeling task predicts whether the home side wins a match.

Target definition:
- `home_win = 1` when home goals > away goals
- `home_win = 0` otherwise

The script applies a chronological train/test split so model evaluation reflects realistic forecasting conditions.

## Feature Selection Choices

### 1. Leakage-Safe Pre-Match Features
Selected features are designed to be known before kickoff or stable tournament context variables, including:
- Team identifiers and venue context (`HomeTeamName`, `AwayTeamName`, `Stadium`, `City`, initials)
- Match timing context (`match_month`, `match_day_of_week`, `is_weekend_match`)
- Match attendance context (`attendance_log1p`)
- Tournament context (`QualifiedTeams`, `tournament_matches_played`, `tournament_goals_scored`, `tournament_attendance`)
- Prior form features (`home_prior_*`, `away_prior_*`, deltas)
- Side-specific winning features (`home_team_prior_home_win_rate`, `away_team_prior_away_win_rate`, `prior_side_win_rate_delta`)
- Attendance-on-winning features (`home_team_prior_win_attendance_avg`, `away_team_prior_win_attendance_avg`, `prior_win_attendance_avg_delta`)
- Team-sheet structure features (`home_starters`, `away_starters`, captains, substitutes, bench size)

### 2. Excluded/Implicitly Not Used
Post-match outcome fields and obvious leakage fields are not selected as predictors for this target. This includes fields such as:
- `HomeTeamGoals`, `AwayTeamGoals`
- `goal_diff`, `total_goals`
- `is_draw`, `away_win`
- Any direct post-result label proxies

### 3. Why This Selection Works
- Retains predictive signal from history, context, and team composition.
- Avoids direct target leakage from match outcomes.
- Keeps feature set interpretable for governance and audit.

## Preprocessing Strategy
The script uses a column-wise preprocessing pipeline:
- Numeric columns:
  - Median imputation
  - Standard scaling
- Categorical columns:
  - Most-frequent imputation
  - One-hot encoding (`handle_unknown='ignore'`)

This standardizes mixed data types and makes model training robust across historical periods.

## Model Candidates
Two baseline classifiers are trained and compared:
- Logistic Regression (balanced class weights)
- Random Forest (balanced class weights, tuned tree constraints)

Both models are wrapped in a shared preprocessing + model pipeline for consistency.

## Evaluation Design

### 1. Temporal Split
Data is first split chronologically (80% train, 20% test), preserving time order.

If that split creates a single-class train or test set, the pipeline falls back to stratified random split to ensure metrics remain meaningful (for example to avoid all-zero precision/recall despite high accuracy).

### 2. Metrics Reported
For each model:
- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC

For the best model, additional diagnostics are saved:
- Confusion matrix
- Full classification report

### 3. Selection Rule
Best model is selected by highest F1 score (balanced precision/recall focus).

## Output Artifacts and Governance
All outputs are written into the `models` governance area:
- Best model artifact (`.joblib`)
- Model comparison summary (`.csv`)
- Best-model metrics report (`.txt`)
- Run logs in `models/logs/`

This supports reproducibility, audits, and longitudinal model performance tracking.

## Interpretation Notes
- Stronger performance from prior-form delta features indicates pre-match momentum is informative.
- Stronger performance from side-specific win-rate features indicates home/away context is informative.
- Stronger performance from win-attendance features indicates crowd context contributes additional signal.
- Venue and team identity one-hot features often capture structural matchups and historical asymmetry.
- Tree-based models may capture nonlinear effects (for example interaction between venue and prior form) better than linear models.

## Recommended Next Enhancements
- Add explicit calibration checks (Brier score, reliability curves).
- Use time-series cross-validation rather than single holdout split.
- Add a draw-specific multiclass model (`home_win`, `draw`, `away_win`).
- Track feature importance and drift across tournament eras.
- Version feature schema and model configs for stricter governance.
