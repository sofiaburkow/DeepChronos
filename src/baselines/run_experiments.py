import subprocess
from itertools import product


classifiers = [
    "multiclass",
    "ensemble",
]

dataset_scenario_opts = [
    # ("darpa2000", "s1_inside"),
    # ("darpa2000", "s1_dmz"),
    ("aitv2", "santos"),
    # ("aitv2", "fox"),
]

feature_group_opts = [
    "aug",
    "reduced",
    "full",
]

subset_opts = [
    # "10b10a",
    # "20b20a",
    # "30b30a",
    # "50b50a",
    # "100b100a",
    # "500b500a",
    # "1000b1000a",
    "balanced",
    # "full"
]

window_opts = [
    10,
    100,
]

# uv run python -m src.baselines.run_experiments
for classifier, (dataset, scenario), feature_group, subset, window_size in product(classifiers, dataset_scenario_opts, feature_group_opts, subset_opts, window_opts):
    
    cmd = [
        "uv", "run", "python", "-m", "src.baselines.train_baseline_lstm",
        "--classifier", str(classifier),
        "--dataset", str(dataset),
        "--scenario", str(scenario),
        "--feature_group", str(feature_group),
        "--subset", str(subset),
        "--window_size", str(window_size),
        "--epochs", str(50),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
