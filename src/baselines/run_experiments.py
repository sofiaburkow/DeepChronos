import subprocess
from itertools import product


classifiers = [
    "ensemble",
    "multiclass",
]

features = [
    "all",
    "sub",
]

window_sizes = [
    10, 
    50, 
    100
]

fractions = [
    1.0, 
    0.5, 
    0.25, 
    0.10, 
    0.05
]

dataset = "aitv2"
scenario = "fox"
epochs = 10

# Generate all combinations
for classifier, feature_group, window_size, fraction in product(
    classifiers, features, window_sizes, fractions
):
    cmd = [
        "uv", "run", "python", "-m", "src.baselines.train_baseline_lstm",
        "--dataset", str(dataset),
        "--scenario", str(scenario),
        "--classifier", str(classifier),
        "--feature_group", str(feature_group),
        "--fraction", str(int(fraction*100)),
        "--window_size", str(window_size),
        "--epochs", str(epochs),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)