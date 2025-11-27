# DeepChronos: Enhancing Data-Driven NIDSs in a Neuro-Symbolic Fashion

This project leverages DeepProbLog to integrate symbolic knowledge from network vulnerabilities into data-driven Network Intrusion Detection Systems (NIDSs), enabling detection of multi-step attacks while improving interpretability.


## Prerequisites

- Python 3.10 (recommended)
- `uv` (Python project/environment manager) installed globally

Install with: 
```bash
pip install uv
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

## Usage

```bash
# Run the main pipeline
uv run python src/train.py

# Or evaluate the model
uv run python src/evaluate.py

# Open a notebook for exploration
uv run jupyter lab notebooks/exploration.ipynb
```


## Project Structure

```csharp
DeepChronos/
│
├── .python-version         # pins Python version for uv/pyenv
├── .gitignore    
├── pyproject.toml          # project metadata and dependencies
├── uv.lock                 # locked dependency versions
├── README.md        
├── data/                   # DARPA 2000 or other datasets
│   └── DARPA_2000/           
│       ├── inside          # Training data      
│       └── dmz             # Test data                       
├── notebooks/              # Jupyter notebooks for exploration and analysis
│   └── exploration.ipynb
├── src/                    # Python source code
│   ├── __init__.py
│   ├── model.py            # DeepProbLog model definition
│   ├── networks.py         # PyTorch neural network modules
│   ├── logic.pl            # ProbLog symbolic rules
│   ├── train.py            # Training pipeline
│   └── evaluate.py         # Evaluation script
└── scripts/
    └── preprocess_darpa.py # preprocessing scripts for DARPA 2000
```