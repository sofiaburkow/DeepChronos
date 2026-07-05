# Feature Engineering

This document describes the feature engineering process for preparing the datasets for model training.

## Available Datasets and Scenarios

For each dataset, the following scenarios are available:
| Dataset | Scenario | Description |
|---------|----------|------------|
| darpa2000 | s1_inside | Inside network traffic |
| darpa2000 | s1_dmz | DMZ network traffic |   
| aitv2 | santos | Santos scenario |
| aitv2 | fox | Fox scenario |

## Generate Statistical Features

The statistical features are generated from the Zeek flow CSV files. It takes the `all_flows_labeled.csv` file as input and computes various statistical features for each flow. The resulting feature file is saved as `all_flows_behavioral.csv` and stored in the `data/interim/<dataset>/<scenario>/flows_labeled/` directory.

How to run the feature generation script:

```bash
uv run python -m src.feature_engineering.statistical_features --dataset <dataset> --scenario <scenario> 
```

## Windowed Features

The windowed features are generated from the statistical feature files. It takes the `all_flows_behavioral.csv` file as input and computes windowed features for each flow based on a specified window size. The resulting windowed feature files are stored in the `data/processed/<dataset>/<scenario>/<feature_group>/windowed/<window_size>/` directory.

In the case where we want to generate windowed features for one scenario, we can run the following command:

```bash
uv run python -m src.feature_engineering.windowing --dataset <dataset> --scenario <scenario>
```

In the case where we want to generate windowed features where we use one scenario for training and another scenario for testing, we can run the following command:

```bash
uv run python -m src.feature_engineering.windowing_cross --dataset <dataset> --scenario <scenario>
```

## Sampling

Finally, the windowed features are sampled to create the final dataset for model training. 

```bash
uv run python -m src.feature_engineering.sample_data --dataset <dataset> --scenario <scenario>
```