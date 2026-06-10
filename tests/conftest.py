from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"


def load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def data_prep_module() -> ModuleType:
    return load_module_from_path("world_cup_data_prep", NOTEBOOKS_DIR / "1_data_prep.py")


@pytest.fixture(scope="session")
def feature_engineering_module() -> ModuleType:
    return load_module_from_path("world_cup_feature_engineering", NOTEBOOKS_DIR / "3_feature_engineering.py")


@pytest.fixture(scope="session")
def model_training_module() -> ModuleType:
    return load_module_from_path(
        "world_cup_model_training",
        NOTEBOOKS_DIR / "4_model_training_and_evaluation.py",
    )


@pytest.fixture
def sample_matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Year": "2010",
                "Datetime": pd.Timestamp("2010-06-12 16:00"),
                "Stage": "Group A",
                "HomeTeamName": "Brazil",
                "AwayTeamName": "Spain",
                "HomeTeamInitials": "BRA",
                "AwayTeamInitials": "ESP",
                "HomeTeamGoals": 2,
                "AwayTeamGoals": 1,
                "HalftimeHomeGoals": 1,
                "HalftimeAwayGoals": 0,
                "Attendance": 50000,
                "MatchID": 1,
            },
            {
                "Year": "2010",
                "Datetime": pd.Timestamp("2010-06-19 16:00"),
                "Stage": "Quarter-finals",
                "HomeTeamName": "Spain",
                "AwayTeamName": "Brazil",
                "HomeTeamInitials": "ESP",
                "AwayTeamInitials": "BRA",
                "HomeTeamGoals": 0,
                "AwayTeamGoals": 0,
                "HalftimeHomeGoals": 0,
                "HalftimeAwayGoals": 0,
                "Attendance": 62000,
                "MatchID": 2,
            },
            {
                "Year": "2014",
                "Datetime": pd.Timestamp("2014-07-13 20:00"),
                "Stage": "Final",
                "HomeTeamName": "Germany",
                "AwayTeamName": "Argentina",
                "HomeTeamInitials": "GER",
                "AwayTeamInitials": "ARG",
                "HomeTeamGoals": 1,
                "AwayTeamGoals": 0,
                "HalftimeHomeGoals": 0,
                "HalftimeAwayGoals": 0,
                "Attendance": 70000,
                "MatchID": 3,
            },
            {
                "Year": "2014",
                "Datetime": pd.Timestamp("2014-07-14 20:00"),
                "Stage": "Third-place",
                "HomeTeamName": "Brazil",
                "AwayTeamName": "Germany",
                "HomeTeamInitials": "BRA",
                "AwayTeamInitials": "GER",
                "HomeTeamGoals": 0,
                "AwayTeamGoals": 2,
                "HalftimeHomeGoals": 0,
                "HalftimeAwayGoals": 1,
                "Attendance": 68000,
                "MatchID": 4,
            },
        ]
    )


@pytest.fixture
def sample_cups_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Year": "2010",
                "Country": "South Africa",
                "Winner": "Spain",
                "QualifiedTeams": 32,
                "MatchesPlayed": 64,
                "GoalsScored": 145,
                "Attendance": 3178856,
            },
            {
                "Year": "2014",
                "Country": "Brazil",
                "Winner": "Germany",
                "QualifiedTeams": 32,
                "MatchesPlayed": 64,
                "GoalsScored": 171,
                "Attendance": 3429873,
            },
        ]
    )
