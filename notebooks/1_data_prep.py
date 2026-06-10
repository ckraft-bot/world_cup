from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

MISSING_TOKENS = {"", "-", "na", "n/a", "null", "none"}

# Schema config is intentionally JSON-like so new datasets can be added declaratively.
SCHEMAS: dict[str, dict[str, Any]] = {
    # first file
    "world_cups": {
        "source_file": "WorldCups.csv",
        "target_file": "world_cups_clean.csv",
        "columns": [
            {"source": "Year", "target": "Year", "type": "str"},
            {"source": "Country", "target": "Country", "type": "str"},
            {"source": "Winner", "target": "Winner", "type": "str"},
            {"source": "Runners-Up", "target": "SecondPlace", "type": "str"},
            {"source": "Third", "target": "ThirdPlace", "type": "str"},
            {"source": "Fourth", "target": "FourthPlace", "type": "str"},
            {"source": "GoalsScored", "target": "GoalsScored", "type": "int"},
            {"source": "QualifiedTeams", "target": "QualifiedTeams", "type": "int"},
            {"source": "MatchesPlayed", "target": "MatchesPlayed", "type": "int"},
            {"source": "Attendance", "target": "Attendance", "type": "int"},
        ],
    },
    # second file
    "world_cup_players": {
        "source_file": "WorldCupPlayers.csv",
        "target_file": "world_cup_players_clean.csv",
        "columns": [
            {"source": "RoundID", "target": "RoundID", "type": "int"},
            {"source": "MatchID", "target": "MatchID", "type": "int"},
            {"source": "Team Initials", "target": "TeamInitials", "type": "str", "uppercase": True},
            {"source": "Coach Name", "target": "CoachName", "type": "str", "uppercase": True},
            {"source": "Line-up", "target": "LineUp", "type": "str"},
            {"source": "Shirt Number", "target": "JerseyNumber", "type": "int"},
            {"source": "Player Name", "target": "PlayerName", "type": "str", "uppercase": True},
            {"source": "Position", "target": "PlayerPosition", "type": "str"},
            {"source": "Event", "target": "Event", "type": "str"},
        ],
    },
    # third file
    "world_cup_matches": {
        "source_file": "WorldCupMatches.csv",
        "target_file": "world_cup_matches_clean.csv",
        "columns": [
            {"source": "Year", "target": "Year", "type": "str"},
            {"source": "Datetime", "target": "Datetime", "type": "datetime"}, # datetime in 24 hour format YYYY-MM-DD HH:MM
            {"source": "Stage", "target": "Stage", "type": "str"},
            {"source": "Stadium", "target": "Stadium", "type": "str"},
            {"source": "City", "target": "City", "type": "str"},
            {"source": "Home Team Name", "target": "HomeTeamName", "type": "str"},
            {"source": "Home Team Goals", "target": "HomeTeamGoals", "type": "int"},
            {"source": "Away Team Goals", "target": "AwayTeamGoals", "type": "int"},
            {"source": "Away Team Name", "target": "AwayTeamName", "type": "str"},
            {"source": "Win conditions", "target": "Winconditions", "type": "str"},
            {"source": "Attendance", "target": "Attendance", "type": "int"},
            {"source": "Half-time Home Goals", "target": "HalftimeHomeGoals", "type": "int"},
            {"source": "Half-time Away Goals", "target": "HalftimeAwayGoals", "type": "int"},
            {"source": "Referee", "target": "Referee", "type": "str", "uppercase": True},
            {"source": "Assistant 1", "target": "Assistant1", "type": "str", "uppercase": True},
            {"source": "Assistant 2", "target": "Assistant2", "type": "str", "uppercase": True},
            {"source": "RoundID", "target": "RoundID", "type": "int"},
            {"source": "MatchID", "target": "MatchID", "type": "int"},
            {"source": "Home Team Initials", "target": "HomeTeamInitials", "type": "str", "uppercase": True},
            {"source": "Away Team Initials", "target": "AwayTeamInitials", "type": "str", "uppercase": True},
        ],
    },
}


def setup_logging() -> None:
    """Configure application logging with loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss,SSS} | {level} | {message}",
    )


def normalize_text(value: str | None) -> str | None:
    """Normalize whitespace and common missing tokens in text fields."""
    if value is None:
        return None

    cleaned = value.strip().replace("\u00a0", " ")
    if cleaned.lower() in MISSING_TOKENS:
        return None
    return cleaned


def normalize_upper_text(value: str | None) -> str | None:
    """Normalize text and convert it to uppercase."""
    cleaned = normalize_text(value)
    if cleaned is None:
        return None
    return cleaned.upper()


def to_int(value: str | None) -> int | None:
    """Parse integer values with support for thousand separators."""
    text = normalize_text(value)
    if text is None:
        return None

    normalized = text.replace(".", "").replace(",", "")
    return int(normalized)


def parse_match_datetime(value: str | None, output_type: str = "date") -> str | None:
    """Parse match datetime values and output standardized date or datetime strings."""
    text = normalize_text(value)
    if text is None:
        return None

    date_formats = ("%d %b %Y - %H:%M", "%d %B %Y - %H:%M")
    for date_format in date_formats:
        try:
            parsed = datetime.strptime(text, date_format)
            if output_type == "datetime":
                return parsed.strftime("%Y-%m-%d %H:%M")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"Unsupported date format: {text}")


def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    """Read a CSV file using a resilient encoding fallback."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with file_path.open(mode="r", newline="", encoding=encoding) as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            logger.info("Loaded {} rows from {} using {}", len(rows), file_path.name, encoding)
            return rows
        except UnicodeDecodeError:
            logger.warning("Failed decoding {} with {}", file_path.name, encoding)

    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {file_path}")


def write_csv_rows(file_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write transformed rows to CSV with UTF-8 encoding."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open(mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote {} rows to {}", len(rows), file_path)


def coerce_value(value: str | None, data_type: str, uppercase: bool = False) -> int | str | None:
    """Convert a raw value to the requested schema type."""
    if data_type == "int":
        return to_int(value)
    if data_type == "date":
        return parse_match_datetime(value, output_type="date")
    if data_type == "datetime":
        return parse_match_datetime(value, output_type="datetime")

    text_value = normalize_text(value)
    if text_value is None:
        return None
    if uppercase:
        return text_value.upper()
    return text_value


def transform_with_schema(rows: list[dict[str, str]], schema: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Transform rows according to a schema definition from SCHEMAS."""
    transformed: list[dict[str, Any]] = []
    columns: list[dict[str, Any]] = schema["columns"]
    fieldnames = [column["target"] for column in columns]

    for row in rows:
        output_row: dict[str, Any] = {}
        for column in columns:
            output_row[column["target"]] = coerce_value(
                value=row.get(column["source"]),
                data_type=column["type"],
                uppercase=bool(column.get("uppercase", False)),
            )
        transformed.append(output_row)

    return transformed, fieldnames


def run_dataset(dataset_name: str) -> None:
    """Run one schema-driven data prep task."""
    schema = SCHEMAS[dataset_name]
    source_path = RAW_DIR / str(schema["source_file"])
    target_path = PROCESSED_DIR / str(schema["target_file"])

    raw_rows = read_csv_rows(source_path)
    cleaned_rows, fieldnames = transform_with_schema(raw_rows, schema)
    write_csv_rows(target_path, cleaned_rows, fieldnames)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for selecting dataset scope."""
    parser = argparse.ArgumentParser(description="Run World Cup data preparation tasks.")
    dataset_choices = list(SCHEMAS.keys()) + ["all"]
    parser.add_argument(
        "--dataset",
        choices=dataset_choices,
        default="all",
        help="Dataset prep task to run.",
    )
    return parser.parse_args()


def run_pipeline(dataset: str) -> None:
    """Dispatch selected data prep tasks."""
    if dataset == "all":
        for task_name in SCHEMAS:
            logger.info("Running task: {}", task_name)
            run_dataset(task_name)
        return

    run_dataset(dataset)


def main() -> None:
    """Application entrypoint."""
    setup_logging()
    args = parse_args()
    run_pipeline(args.dataset)


if __name__ == "__main__":
    main()
