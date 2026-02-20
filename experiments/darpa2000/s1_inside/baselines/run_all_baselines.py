import subprocess
from itertools import product

# Models to evaluate
models = [
    #"src.baselines.train_ensemble_lstm",
    "src.baselines.train_multi_class_lstm",
]

# Hyperparameters
# window_sizes = [10, 50, 100]
window_sizes = [10]
resampled_opts = [False, True]
class_weight_opts = [False, True]

# Generate all combinations
for model, window_size, resampled, class_weights in product(
    models, window_sizes, resampled_opts, class_weight_opts
):
    cmd = [
        "uv", "run", "python", "-m", model,
        "--window_size", str(window_size),
    ]

    if resampled:
        cmd.append("--resampled")

    if class_weights:
        cmd.append("--class_weights")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
