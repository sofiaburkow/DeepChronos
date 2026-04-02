import subprocess
from itertools import product

# Models to evaluate
classifier_opts = [
    "ensemble",
    "multi_class",
]

features_opts = [
    "all",
    "sub",
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
    True,
]

# Generate all combinations
for classifier, feature_group, dataset_variant, window_size, class_weights in product(
    classifier_opts, features_opts, dataset_variant_opts, window_opts, class_weight_opts
):
    cmd = [
        "uv", "run", "python", "-m", "src.baselines.train_baseline_lstm",
        "--classifier", str(classifier),
        "--feature_group", str(feature_group),
        "--dataset_variant", str(dataset_variant),
        "--window_size", str(window_size),
    ]

    if class_weights:
        cmd.append("--class_weights")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)