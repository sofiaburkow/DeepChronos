# Feature Engineering

This document describes how to process the datasets for model training.


## Dataset Options

**dataset:** 
- either `darpa2000` or `aitv2`.

**scenario:** 
- e.g., `santos` (AIT-LDS V2) or `s1_inside` (DARPA 2000). 
- if cross-scenario experiments are desired, the `scenario` argument can be `santos_fox` (AIT-LDS V2) or `s1_inside_s1_dmz` (DARPA 2000). In this case, the first scenario is used for training and the second scenario is used for testing.


## Processing Steps

### 1. Generate Statistical Features

The statistical features are generated from the Zeek flow CSV files. It takes the `all_flows_labeled.csv` file as input and computes various statistical features for each flow. 

The statistical features are computed over a specified time window (e.g., 60 seconds). This can be adjusted in the `src/feature_engineering/statistical_features.py` script.

Run the following command to execute the feature generation script:

```bash
uv run python -m src.feature_engineering.statistical_features --dataset <dataset> --scenario <scenario> 
```

The resulting file is saved to `data/interim/<dataset>/<scenario>/flows_labeled/all_flows_behavioral.csv`.

NB: this step might take a while to complete, depending on the size of the dataset and the number of flows.

### 2. Create Windowed Datasets

To train the LSTMs on time-series data, we need to create windowed datasets. The windowed dataset is generated from the statistical feature files. It takes the `all_flows_statistical.csv` file as input and computes windowed features for each flow based on a specified feature group and window size. Uncomment the desired feature group and window size in the script.

In the case where we want to generate windowed features for one scenario (e.g., `santos`), we can run the following command:

```bash
uv run python -m src.feature_engineering.windowing --dataset <dataset> --scenario <scenario>
```

In case of cross-scenario experiments, we can run the following command:

```bash
uv run python -m src.feature_engineering.windowing_cross --dataset <dataset> --scenario <scenario>
```

The resulting files are stored in the `data/processed/<dataset>/<scenario>/<feature_group>/windowed/<window_size>/` directory.

### 3. Sample Data

In order to test the models ability to learn from limited data, we sub-sample the dataset. Similar to the previous step, uncomment the desired feature group and window size in the script.

Run the following command to execute the sampling script:

```bash
uv run python -m src.feature_engineering.sample_data --dataset <dataset> --scenario <scenario>
```