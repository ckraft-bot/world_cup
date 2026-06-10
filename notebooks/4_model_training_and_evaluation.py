from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized project paths for model training/evaluation outputs."""

    base_dir: Path
    features_dir: Path
    models_dir: Path
    logs_dir: Path


from pathlib import Path

def resolve_paths() -> ProjectPaths:
    """Resolve repository-relative paths regardless of run location."""
    base_dir = Path(__file__).resolve().parents[1]

    features_dir = base_dir / "data" / "processed" / "features"
    models_dir = base_dir / "models"
    logs_dir = models_dir / "logs"

    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        base_dir=base_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
    )

def configure_logging(logs_dir: Path) -> Path:
    """Configure console and file logging for traceable model governance."""
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"model_training_{run_stamp}.log"

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO")
    logger.add(log_path, level="INFO", encoding="utf-8")

    return log_path


def load_feature_table(features_dir: Path) -> pd.DataFrame:
    """Load country-year engineered features for winner-likelihood modeling."""
    file_path = features_dir / "country_tournament_features.csv"
    df = pd.read_csv(file_path)
    df = df.sort_values(["Year", "Team"]).reset_index(drop=True)
    logger.info("Loaded feature table: {} | shape={}", file_path, df.shape)
    return df


def select_country_winner_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select country-level features aligned to winner likelihood modeling."""
    selected = [
        "Year",
        "Team",
        "HostCountry",
        "QualifiedTeams",
        "tournament_matches_played",
        "tournament_attendance",
        "matches_total",
        "wins_total",
        "draws_total",
        "losses_total",
        "goals_for_total",
        "goals_against_total",
        "goal_balance_total",
        "win_rate_total",
        "home_matches",
        "home_wins",
        "home_win_rate",
        "away_matches",
        "away_wins",
        "away_win_rate",
        "home_away_win_rate_delta",
        "avg_attendance",
        "home_avg_attendance",
        "away_avg_attendance",
        "win_attendance_avg",
        "home_win_attendance_avg",
        "away_win_attendance_avg",
        "attendance_win_minus_avg",
        "home_away_win_attendance_delta",
        "is_host_team",
    ]

    available = [col for col in selected if col in df.columns]
    missing = sorted(set(selected) - set(available))
    if missing:
        logger.warning("Missing selected columns skipped: {}", missing)

    return available


def split_train_test_by_year(
    df: pd.DataFrame,
    target_col: str,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by tournament year and fallback to stratified rows if class diversity is poor."""
    years = sorted(df["Year"].dropna().unique().tolist())
    if len(years) < 3:
        raise ValueError("Not enough unique years for year-based split.")

    test_year_count = max(1, int(round(len(years) * test_ratio)))
    test_years = set(years[-test_year_count:])

    train_df = df[~df["Year"].isin(test_years)].copy()
    test_df = df[df["Year"].isin(test_years)].copy()

    train_classes = train_df[target_col].nunique(dropna=False)
    test_classes = test_df[target_col].nunique(dropna=False)

    if train_classes < 2 or test_classes < 2:
        logger.warning(
            "Year-based split has limited class diversity (train_classes={}, test_classes={}). "
            "Falling back to stratified random split for meaningful evaluation.",
            train_classes,
            test_classes,
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=42,
            stratify=df[target_col],
        )
        train_df = train_df.sort_values(["Year", "Team"]).reset_index(drop=True)
        test_df = test_df.sort_values(["Year", "Team"]).reset_index(drop=True)

    logger.info("Train rows: {} | Test rows: {}", len(train_df), len(test_df))
    logger.info("Train year range: {}-{}", train_df["Year"].min(), train_df["Year"].max())
    logger.info("Test year range: {}-{}", test_df["Year"].min(), test_df["Year"].max())

    return train_df, test_df


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    logger.info("Numeric features: {} | Categorical features: {}", len(numeric_cols), len(categorical_cols))
    return preprocessor


def evaluate_binary_classifier(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate binary classifier and return metrics dictionary."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    if y_test.nunique(dropna=False) < 2:
        roc_auc: float = float("nan")
        logger.warning("ROC AUC is undefined because y_test contains a single class.")
    else:
        roc_auc = float(roc_auc_score(y_test, y_proba))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics


def train_and_compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> tuple[str, Pipeline, pd.DataFrame]:
    """Train baseline models, compare metrics, and return best model."""
    model_specs: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    records: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(x_train, y_train)
        metrics = evaluate_binary_classifier(pipeline, x_test, y_test)

        records.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        fitted_models[model_name] = pipeline

        logger.info("{} metrics: {}", model_name, records[-1])

    comparison = pd.DataFrame(records).sort_values("f1", ascending=False).reset_index(drop=True)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted_models[best_name]

    return best_name, best_model, comparison


def persist_outputs(
    paths: ProjectPaths,
    best_name: str,
    best_model: Pipeline,
    comparison: pd.DataFrame,
    best_metrics: dict[str, Any],
) -> None:
    """Save model artifact and evaluation outputs for governance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = paths.models_dir / f"country_winner_{best_name}_{timestamp}.joblib"
    summary_path = paths.models_dir / f"country_winner_model_comparison_{timestamp}.csv"
    metrics_path = paths.models_dir / f"country_winner_{best_name}_metrics_{timestamp}.txt"

    joblib.dump(best_model, model_path)
    comparison.to_csv(summary_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Best model: {best_name}\n\n")
        for key, value in best_metrics.items():
            handle.write(f"{key}: {value}\n")

    logger.info("Saved model artifact: {}", model_path)
    logger.info("Saved model comparison: {}", summary_path)
    logger.info("Saved best-model metrics: {}", metrics_path)


def main() -> None:
    """Train and evaluate country-level World Cup winner likelihood models."""
    paths = resolve_paths()
    log_path = configure_logging(paths.logs_dir)

    df = load_feature_table(paths.features_dir)
    feature_cols = select_country_winner_feature_columns(df)

    if "is_world_cup_winner" not in df.columns:
        raise ValueError("Target column 'is_world_cup_winner' not found in country_tournament_features.csv")

    modeling_df = df.dropna(subset=["is_world_cup_winner"]).copy()
    modeling_df["is_world_cup_winner"] = modeling_df["is_world_cup_winner"].astype(int)

    train_df, test_df = split_train_test_by_year(
        modeling_df,
        target_col="is_world_cup_winner",
        test_ratio=0.2,
    )

    x_train = train_df[feature_cols]
    y_train = train_df["is_world_cup_winner"]
    x_test = test_df[feature_cols]
    y_test = test_df["is_world_cup_winner"]

    preprocessor = build_preprocessor(x_train)

    best_name, best_model, comparison = train_and_compare_models(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    best_metrics = evaluate_binary_classifier(best_model, x_test, y_test)

    persist_outputs(
        paths=paths,
        best_name=best_name,
        best_model=best_model,
        comparison=comparison,
        best_metrics=best_metrics,
    )

    logger.info("Training/evaluation completed. Best model: {}", best_name)
    logger.info("Run log: {}", log_path)


if __name__ == "__main__":
    main()
