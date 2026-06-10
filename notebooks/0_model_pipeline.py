from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure simple console logging for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_paths() -> tuple[Path, Path]:
    """Resolve project and workspace paths regardless of invocation location."""
    notebooks_dir = Path(__file__).resolve().parent
    project_dir = notebooks_dir.parent
    workspace_root = project_dir.parent.parent
    return project_dir, workspace_root


def run_command(command: Sequence[str], cwd: Path, step_name: str) -> None:
    """Run a subprocess command and stream logs to the terminal."""
    printable = " ".join(command)
    LOGGER.info("Starting step: %s", step_name)
    LOGGER.info("Command: %s", printable)

    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {step_name} (exit code {result.returncode}). Command: {printable}"
        )

    LOGGER.info("Completed step: %s", step_name)


def run_pipeline(skip_eda: bool = False) -> None:
    """Run the World Cup workflow in order."""
    project_dir, workspace_root = resolve_paths()
    notebooks_rel = project_dir.relative_to(workspace_root)

    script_data_prep = str(notebooks_rel / "notebooks" / "1_data_prep.py")
    notebook_eda = str(notebooks_rel / "notebooks" / "2_eda.ipynb")
    script_features = str(notebooks_rel / "notebooks" / "3_feature_engineering.py")
    script_model = str(notebooks_rel / "notebooks" / "4_model_training_and_evaluation.py")

    run_command(
        command=[sys.executable, script_data_prep],
        cwd=workspace_root,
        step_name="1_data_prep.py",
    )

    if not skip_eda:
        run_command(
            command=[
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=-1",
                notebook_eda,
            ],
            cwd=workspace_root,
            step_name="2_eda.ipynb",
        )
    else:
        LOGGER.info("Skipping step: 2_eda.ipynb")

    run_command(
        command=[sys.executable, script_features],
        cwd=workspace_root,
        step_name="3_feature_engineering.py",
    )

    run_command(
        command=[sys.executable, script_model],
        cwd=workspace_root,
        step_name="4_model_training_and_evaluation.py",
    )

    LOGGER.info("Pipeline finished successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for pipeline execution."""
    parser = argparse.ArgumentParser(description="Run World Cup notebooks pipeline in order.")
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip executing 2_eda.ipynb.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    configure_logging()
    args = parse_args()

    try:
        run_pipeline(skip_eda=args.skip_eda)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
