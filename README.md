# DeepChronos  

Enhancing Data-Driven NIDSs in a Neuro-Symbolic Fashion

DeepChronos integrates symbolic reasoning into data-driven Network Intrusion Detection Systems (NIDSs) to improve multi-step attack detection.

The project combines:

- LSTM-based flow classification
- Structured attack progression knowledge
- DeepProbLog neuro-symbolic reasoning

The goal is to enhance detection consistency and interpretability for multi-phase cyber attacks.


## Overview

Traditional NIDS models operate purely at the flow level and may fail to capture structured attack progression.

DeepChronos augments neural classifiers with symbolic constraints encoding multi-step attack logic, enabling:

- Phase-aware reasoning
- Structured attack detection
- Improved consistency across attack stages


## Repository Structure

See:

→ `docs/project_structure.md`


## Installation

### Requirements

- Python 3.10
- `uv` (Python project manager)
- SWI-Prolog (required by DeepProbLog)

Install `uv`:

```bash
pip install uv
```

Install SWI-Prolog (Ubuntu):

```bash
sudo apt update
sudo apt install swi-prolog
```

Ensure `uv` can find `swipl` in your PATH:

```bash
uv run which swipl
```


### Setup

```bash
# Clone the repository
git clone git@github.com:sofiaburkow/DeepChronos.git
cd DeepChronos

# Initialize the project environment (uv automatically creates a .venv)
uv install

# Pin Python version for reproducibility
uv python pin 3.10

# Sync dependencies and prepare environment
uv sync
```


## Data Preprocessing

The project uses the DARPA 2000 dataset.

Full preprocessing instructions (PCAP → Zeek → labeled flows):

→ docs/preprocessing.md

Dataset description:

→ docs/datasets.md


## Running Experiments

Reproducibility instructions:

→ docs/experiments.md

Experiments are organized under:

```bash
experiments/darpa2000/s1_inside/
```

