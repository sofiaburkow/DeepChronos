import subprocess
from itertools import product

dataset_scenario_opts = [
    ("darpa2000", "s1_inside"),
    ("darpa2000", "s1_dmz"),
    ("aitv2", "santos"),
    ("aitv2", "fox"),
]

feature_group_opts = [
    "full",
    "reduced",
    "behavioral"
]

window_opts = [
    10,
    100,
]

# uv run python -m src.deepproblog.pretrain_all
for (dataset, scenario), feature_group, window_size in product(dataset_scenario_opts, feature_group_opts, window_opts):
    
    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.pretrain",

        "--dataset", str(dataset),
        "--scenario", str(scenario),
        "--feature_group", str(feature_group),
        "--window_size", str(window_size),
        "--epochs", str(10),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
