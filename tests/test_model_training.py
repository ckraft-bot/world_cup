from __future__ import annotations

import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def make_model_df() -> pd.DataFrame:
    rows = []
    teams = ["A", "B", "C", "D"]
    for year in [2006, 2010, 2014, 2018, 2022]:
        for idx, team in enumerate(teams):
            rows.append(
                {
                    "Year": year,
                    "Team": team,
                    "feature_num": float(year % 10 + idx),
                    "feature_cat": "host" if idx == 0 else "guest",
                    "is_world_cup_winner": 1 if idx == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_split_train_test_by_year_returns_disjoint_sets(model_training_module) -> None:
    df = make_model_df()
    train_df, test_df = model_training_module.split_train_test_by_year(
        df,
        target_col="is_world_cup_winner",
        test_ratio=0.2,
    )

    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["Year"]).isdisjoint(set(test_df["Year"]))


def test_build_preprocessor_detects_numeric_and_categorical(model_training_module) -> None:
    x_train = pd.DataFrame(
        {
            "Year": [2010, 2014],
            "Team": ["Spain", "Germany"],
            "win_rate_total": [0.8, 0.75],
        }
    )

    preprocessor = model_training_module.build_preprocessor(x_train)
    transformers = {name: cols for name, _, cols in preprocessor.transformers}

    assert "num" in transformers
    assert "cat" in transformers
    assert set(transformers["num"]) == {"Year", "win_rate_total"}
    assert set(transformers["cat"]) == {"Team"}


def test_evaluate_binary_classifier_handles_single_class_y_test(model_training_module) -> None:
    x_train = pd.DataFrame(
        {
            "Year": [2006, 2010, 2014, 2018],
            "Team": ["A", "B", "C", "D"],
            "value": [0.2, 0.3, 0.8, 0.9],
        }
    )
    y_train = pd.Series([0, 0, 1, 1])

    preprocessor = model_training_module.build_preprocessor(x_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)

    x_test = x_train.iloc[[0, 1]].copy()
    y_test = pd.Series([0, 0])

    metrics = model_training_module.evaluate_binary_classifier(model, x_test, y_test)

    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}.issubset(metrics)
    assert math.isnan(metrics["roc_auc"])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
