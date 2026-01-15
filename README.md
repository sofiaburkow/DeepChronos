# DeepChronos: Enhancing Data-Driven NIDSs in a Neuro-Symbolic Fashion

This project leverages DeepProbLog to integrate symbolic knowledge from network vulnerabilities into data-driven Network Intrusion Detection Systems (NIDSs), enabling detection of multi-step attacks while improving interpretability.


## Prerequisites

- Python 3.10 (recommended)
- `uv` (Python project/environment manager) installed globally

Install with: 
```bash
pip install uv
```

- `swi-prolog` installed (for DeepProbLog)

On Ubuntu:
```bash
sudo apt update
sudo apt install swi-prolog
```
Ensure `uv` can find `swipl` in your PATH.
```bash
uv run which swipl
```

## Setup Instructions

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