import subprocess

dataset_scenario_opts = [
    ("darpa2000", "s1_inside"),
    # ("aitv2", "santos"),
]

feature_group = "base"
subset = "full"
# window_size = 100
window_size = 10
learning_rate = 1e-3
epochs = 50

# uv run python -m src.deepproblog.pretrain_all
for (dataset, scenario) in dataset_scenario_opts:
    
    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.pretrain",

        "--dataset", str(dataset),
        "--scenario", str(scenario),

        "--feature_group", str(feature_group),
        "--subset", str(subset),
        "--window_size", str(window_size),
        "--learning_rate", str(learning_rate),
        "--epochs", str(epochs),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
