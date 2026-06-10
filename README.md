## Overview

This project builds country-level World Cup winner likelihood models from historical FIFA World Cup data.

Current target:
- `is_world_cup_winner = 1` if a country wins the tournament in that year, else `0`.

## Testing

The project includes a baseline `pytest` suite in `tests/` focused on:

- Data prep parsing/coercion rules.
- Feature engineering correctness on synthetic inputs.
- Model utility behavior for splitting, preprocessing, and metric generation.
- A lightweight feature pipeline smoke test.

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Run tests

```powershell
pytest -q
```

Run tests and save a timestamped log file in `test-results/`:

```powershell
python run_tests.py
```

## Pipeline

Run the full workflow in order:
- data preparation
- EDA notebook execution
- feature engineering
- model training and evaluation

```powershell
python notebooks/pipeline.py
```

Skip notebook execution when you only want script stages:

```powershell
python notebooks/pipeline.py --skip-eda
```

