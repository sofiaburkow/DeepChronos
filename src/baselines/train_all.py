import subprocess
from itertools import product

# Models to evaluate
model_opts = [
    "src.baselines.train_multi_class_lstm",
    "src.baselines.train_ensemble_lstm",
]

features_opts = [
    "all",
    "dpl",
]

# Hyperparameters
window_opts = [
    10, 
    50, 
    100
    ]

dataset_variant_opts = [
    "original",
    "up", 
    "down",
    ]

class_weight_opts = [
    False, 
    True
]

# Generate all combinations
for model, window, dataset_variant, class_weights in product(
    model_opts, window_opts, dataset_variant_opts, class_weight_opts
):
    cmd = [
        "uv", "run", "python", "-m", model,
        "--window_size", str(window),
        "--dataset_variant", str(dataset_variant)
    ]

    if class_weights:
        cmd.append("--class_weights")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
