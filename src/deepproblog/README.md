# Training DeepProbLog Models

DeepProbLog models are trained using the `train.py` script, which provides a range of command-line arguments for configuring the training process.

To simplify reproducing the experiments, the repository includes several helper scripts in the `experiments/` directory. Each script defines one or more experimental configurations; uncomment the desired lines to run a specific experiment.

## Running Scripts

All Python scripts should be executed as modules from the project root directory.

For example, to run the generalizability study:

```bash
python -m src.deepproblog.experiments.generalizability_study
```

## Pre-training

Models can optionally be initialized using pre-trained weights before DeepProbLog training.

Use the `pretrain.py` script to pre-train a model with custom command-line options, or run `pretrain_all.py` to pre-train models for all supported datasets and scenarios.
