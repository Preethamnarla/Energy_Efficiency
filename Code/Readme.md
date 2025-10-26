# Energy Efficient — Code

This repository contains the code, configs and notebooks used for experiments, models and utilities related to the Energy Efficient project.

Maintainer: Preetham Narla <narla.preetham@gmail.com>

## Purpose
Self-contained code for:
- data preprocessing and feature engineering
- model training and evaluation
- analysis and visualization (notebooks)
- reproducible experiments (configs & scripts)

## Project layout (adjust if needed)
```
/data/                      # raw and processed data (not included)
/configs/
    └─ default.yaml           # example experiment config
/src/
    ├─ train.py               # training entrypoint
    ├─ evaluate.py            # evaluation script
    └─ data_preprocessing.py  # data cleaning & feature engineering
/notebooks/
    └─ analysis.ipynb         # EDA and reproducible analysis
/scripts/
    ├─ run_experiment.sh
    └─ run_experiment.bat
/results/                   # checkpoints, logs, metrics (gitignored)
/tests/                     # unit tests
/requirements.txt
/README.md
/LICENSE
```

## Prerequisites
- Python 3.8+
- pip
- Optional: GPU + CUDA for deep models

## Quick setup
1. Create and activate virtual environment
     - Windows (from project root): python -m venv .venv && .venv\Scripts\activate
     - macOS/Linux: python -m venv .venv && source .venv/bin/activate
2. Install dependencies:
     - pip install -r requirements.txt

## Common commands
- Train (example):
    - python src/train.py --config configs/default.yaml --output results/experiment1
- Evaluate:
    - python src/evaluate.py --model-path results/experiment1/checkpoint.pt --data data/processed/test.csv
- Prepare data:
    - python src/data_preprocessing.py --input data/raw --output data/processed
- Run notebooks:
    - jupyter lab notebooks/analysis.ipynb

Modify arguments and paths to match your local config files.

## Data
- Place raw datasets in data/raw and processed outputs in data/processed.
- Ensure data_preprocessing.py reproduces processing steps used in experiments.

## Experiments & reproducibility
- Keep experiment hyperparameters in configs/*.yaml
- Save checkpoints, logs and metrics to results/<experiment-name> to reproduce runs
- Add a short README in each results/<experiment-name>/ with the config and command used

## Tests
- Add unit tests under /tests and run:
    - pytest -q

## Contributing
- Open an issue to discuss major changes
- Create feature branches, run tests, and submit PRs with a clear description

## License
Add a LICENSE file (e.g., MIT) at repo root.

## TODO
- Add example dataset and minimal config for quickstart
- Document expected data schema and column names

Contact: Preetham Narla — narla.preetham@gmail.com