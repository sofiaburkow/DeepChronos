# DARPA Data & Training Instructions

This README explains the minimal steps to: 
1. process the raw DARPA flows into windowed numpy datasets, 
2. pretrain per-phase LSTM models, and 
3. train / evaluate the DeepProbLog (DPL) multi-step model that uses the pretrained LSTMs.

## 1. Process Data

The data preparation script converts the raw flows CSV into windowed numpy arrays saved under `src/DARPA/data/processed/`.

Usage (from repository root):

```bash
uv run python src/DARPA/data/process_data.py
```

Notes:
 - The script builds sliding windows (default window size in the script). If you need a different window size, edit the `window_size` variable in the script.
 - Both original and resampled datasets are created. The resampled dataset balances the classes using RandomOverSampler. By default, both datasets are saved under `src/DARPA/data/processed/original/` and `src/DARPA/data/processed/resampled/`.

## 2. Pretrain Per-Phase LSTMs (if needed)

Once the processed data is available, run the pretraining script to train and save pretrained LSTMs for each attack phase. By default the pretrained models are saved under `src/DARPA/pretrained/`.

```bash
uv run python src/DARPA/pretrained/create_pretrained.py
```

You can also pass `--dataset_dir` and `--out` to the script to override locations, for example:

```bash
uv run python src/DARPA/pretrained/create_pretrained.py --dataset_dir src/DARPA/data/processed --out src/DARPA/pretrained
```

## 3. Train DeepProbLog Models

There are several options for training the DPL multi-step model:
- The `--function_name` argument specifies which attack function to train on (e.g., `ddos`, `portscan`, etc.).
- The `--pretrained` flag indicates whether to use the pretrained LSTMs from step 2.
- The `--resampled` flag indicates whether to use the resampled dataset.
- The `--lookback_limit` argument sets the maximum number of previous time steps to consider when building the multi-step history.

Example usage:
```bash
uv run python src/DARPA/multi_step.py --function_name ddos --pretrained --resampled --lookback_limit --seed 123
```

The resulting DPL model snapshots and logs will be saved under `src/DARPA/results/`.