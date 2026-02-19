# Project Structure

This document describes the organization of the DeepChronos repository.

The project follows a modular structure separating:

- Data
- Source code
- Experiments
- Documentation

---

## Top-Level Overview

DeepChronos/
│
├── data/
├── docs/
├── experiments/
├── src/
├── main.py
├── pyproject.toml
└── README.md


---

## `data/`

Contains all dataset-related files.

data/
├── raw/ → Original PCAPs and unprocessed DARPA files
├── interim/ → Intermediate outputs (e.g., Zeek logs, partial CSVs)
└── processed/ → Final labeled flow datasets used for training

- `raw/` should never be modified.
- `processed/` contains the final labeled datasets used by models.

Detailed preprocessing steps are described in:
→ `docs/preprocessing.md`

---

## `docs/`

Project documentation.

docs/
├── datasets.md
├── preprocessing.md
├── experiments.md
└── project_structure.md


- `datasets.md` – Description of DARPA 2000 and dataset splits
- `preprocessing.md` – Full pipeline from PCAP → Zeek → labeled flows
- `experiments.md` – Reproducibility guide for all experiments
- `project_structure.md` – This file

---

## `experiments/`

Stores experiment outputs and configuration results.

experiments/
└── darpa2000/
└── s1_inside/
├── baselines/
└── deepproblog/

This directory contains:
- Trained model checkpoints
- Logs
- Evaluation results
- Plots

It is separated from `src/` to clearly distinguish:
- **Code**
- **Experimental artifacts**

---

## `src/`

Contains all source code.

src/
├── baselines/
├── datasets/
├── deepproblog/
├── evaluation/
├── feature_engineering/
├── flow_processing/
└── networks/

### `baselines/`
Pure neural models (e.g., LSTM classifiers).

### `datasets/`
PyTorch dataset definitions and data loading utilities.

### `deepproblog/`
Neuro-symbolic models and logic programs integrating DeepProbLog.

### `evaluation/`
Metrics, confusion matrices, and evaluation utilities.

### `feature_engineering/`
Feature extraction and transformation logic.

### `flow_processing/`
Flow-level data manipulation and preparation.

### `networks/`
Reusable neural network architectures (e.g., LSTM variants).

---

## `main.py`

Entry point for running experiments via CLI.

---

## `pyproject.toml`

Defines:
- Project dependencies
- Python version
- Build configuration

Environment management is handled using `uv`.

---

## Design Philosophy

The repository is structured to ensure:

- Clear separation between data, models, and experiments
- Reproducibility of results
- Modular extensibility for additional attack scenarios
- Clean distinction between neural and neuro-symbolic components

