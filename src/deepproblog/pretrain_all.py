import subprocess

dataset_opts = [
    ("darpa2000", "s1_inside", 10),
    ("darpa2000", "s1_inside_s1_dmz", 10),
    ("darpa2000", "s1_dmz", 10),

    # ("aitv2", "santos", 100),
    # ("aitv2", "santos_fox", 100),
    # ("aitv2", "fox", 100),
]

feature_group = "base"
subset = "full"
learning_rate = 1e-3
epochs = 50

# uv run python -m src.deepproblog.pretrain_all
for (dataset, scenario, window_size) in dataset_opts:
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
