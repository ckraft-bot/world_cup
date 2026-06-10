# World Cup Modeling Conclusion

## Project Outcome
This project delivers an end-to-end pipeline from raw FIFA World Cup data to trained country-level winner-likelihood models.

The modeling target is:
- is_world_cup_winner = 1 when the country won the tournament in that year.
- is_world_cup_winner = 0 otherwise.

## Feature Engineering Summary
The workflow produces three core feature tables:
- data/processed/features/match_features.csv
- data/processed/features/team_year_features.csv
- data/processed/features/country_tournament_features.csv

The final training table includes country-year features for:
- Tournament context (host country, qualified teams, tournament attendance, matches played).
- Team outcome aggregates (wins/draws/losses, goals for/against, goal balance, win rate).
- Home vs away performance deltas.
- Attendance behavior (overall and win-specific attendance signals).

## Models and Preprocessing
Two baseline classifiers are trained using a shared preprocessing stack:
- logistic_regression (class_weight=balanced)
- random_forest (class_weight=balanced)

Preprocessing:
- Numeric columns: median imputation + standard scaling.
- Categorical columns: most-frequent imputation + one-hot encoding.

## Latest Evaluation Summary
Latest artifacts from the current pipeline run:
- models/country_winner_model_comparison_20260609_175342.csv
- models/country_winner_logistic_regression_metrics_20260609_175342.txt
- models/country_winner_logistic_regression_20260609_175342.joblib

Latest comparison metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.9535 | 0.3750 | 0.7500 | 0.5000 | 0.9840 |
| random_forest | 0.9380 | 0.3000 | 0.7500 | 0.4286 | 0.9740 |

Model selection result:
- Best model by F1 score: logistic_regression.

Interpretation:
- Ranking quality is strong (high ROC AUC).
- Precision remains modest due to severe class imbalance (few winners vs many non-winners).

Top logistic-regression coefficients (latest model):
- Strongest positive (overall): `cat__Team_Germany FR` = +1.0149
- Strongest positive (general performance signal): `num__wins_total` = +0.8794
- Next strong positive: `num__home_wins` = +0.8198
- Strongest negative: `num__losses_total` = -1.1143

Interpretation notes:
- In this small historical dataset, country identity features can have large coefficients.
- Among reusable performance features, `wins_total` is stronger than `home_wins`, and `losses_total` is the strongest negative signal.

## Validation and Testing Status
Operational validation completed:
- The orchestrated pipeline runs successfully in order (data prep, feature engineering, training/evaluation).
- New model artifacts and logs are generated for each run.

Automated tests are now available in tests/ and passing:
- Data prep parsing/coercion tests.
- Feature engineering unit tests.
- Model utility tests.
- Integration smoke test.

## Final Conclusion
This project is in a stable baseline state for country-level World Cup winner-likelihood modeling. The pipeline is reproducible, test-backed, and logistic regression is currently the best baseline under F1.

The next gains are most likely to come from:
- Improved handling of rare positive class outcomes.
- Threshold tuning and probability calibration.
- Time-aware validation and additional tournament-level predictive signals.

In summary, I hypothesized that home_wins and attendance would be the strongest predictors. The model only partially supports this hypothesis: home_wins is a strong positive predictor, but not the strongest. The strongest overall positive predictor is the team identity feature Team_Germany FR (+1.0149). Among reusable features, wins_total (+0.8794) ranks above home_wins (+0.8198), while losses_total (-1.1143) is the strongest negative predictor. Attendance-based signals were included, but they were not among the top coefficients in this run.
