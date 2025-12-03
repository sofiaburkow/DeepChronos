# ML Experiments

## Experimental Setup

There are three variables that we want to experiment with:
- Different models
    - Decision Tree
    - Random Forest
    - SVMs 
    - MLPs
- Different feature sets
    - Based on flow features extracted from zeek conn logs
- Different training data splits 
    1. Inside data only (60% train, 20% val, 20% test)
    2. Train on inside, test on DMZ
    3. Train on inside data (Scenario 1), test on inside data (Scenario 2)

In the beginning, only binary classification: attack vs benign

## Features

Run the following script to create ML-ready feature matrices from labeled DARPA 2000 flow CSVs:
```bash
uv run python experiments/features/build_features.py
```

This script:
- loads labeled flow data
- selects useful flow-level features
- encodes categorical fields (proto, service, conn_state)
- encodes or drops IP addresses (configurable)
- normalizes numeric features
- produces X (features) and y (labels)
- saves them to disk for downstream ML experiments

Usage:
```bash
uv run python experiments/features/build_features.py \
    --input data/DARPA_2000/inside/inside_labeled_flows_all.csv \
    --output_dir experiments/features/output \
    --ip-encoding none
```