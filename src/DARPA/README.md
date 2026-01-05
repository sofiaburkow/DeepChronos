# DARPA data & training quickstart

This README explains the minimal steps to: (1) process the raw DARPA flows into windowed numpy datasets, (2) pretrain per-phase LSTM models, and (3) train / evaluate the DeepProbLog (DPL) multi-step model that uses the pretrained LSTMs.

## 1. Process the data

The data preparation script converts the raw flows CSV into windowed numpy arrays saved under `src/DARPA/data/processed/`.

Usage (from repository root):

```bash
uv run python src/DARPA/data/process_data.py
```

Notes:
 - The script expects a features list (default: `src/DARPA/data/features.json`) and a raw CSV (`src/DARPA/data/raw/flows.csv`).
 - Output files (example): `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `y_phase_<i>_train.npy`, etc.
 - The script builds sliding windows (default window size in the script). If you need a different window size, edit the `window_size` variable in the script.

## 2. Pretrain per-phase LSTMs

Once the processed data is available, run the pretraining script to train and save pretrained LSTMs for each attack phase. By default the pretrained models are saved under `src/DARPA/pretrained/`.

```bash
uv run python src/DARPA/pretrained/create_pretrained.py
```

You can also pass `--dataset_dir` and `--out` to the script to override locations, for example:

```bash
uv run python src/DARPA/pretrained/create_pretrained.py --dataset_dir src/DARPA/data/processed --out src/DARPA/pretrained
```

## 3. Train / run the DeepProbLog multi-step model

After you have pretrained LSTMs saved (one per phase), the `src/DARPA/multi_step.py` script will load those pretrained weights and run the DPL experiment (builds the model, sets the engine, and runs training/evaluation).

```bash
uv run python src/DARPA/multi_step.py
```

## Where things are saved

- Processed data: `src/DARPA/data/processed/`
- Pretrained LSTMs: `src/DARPA/pretrained/` (models and results)