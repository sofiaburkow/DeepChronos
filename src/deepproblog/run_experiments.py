import subprocess
from itertools import product

# dataset, scenario, logic_file
dataset_opts = [
    # ("darpa2000", "s1_inside", "darpa"),
    # ("darpa2000", "s1_inside", "darpa_flags"),
    ("aitv2", "santos", "ait_logic"),
    # ("aitv2", "santos", "ait_flags"),
]

feature_group_opts = [
    "aug",
    "full",
    "reduced",
]

subset_opts = [
    "balanced",
    # "10b10a",
    # "20b20a",
    # "30b30a",
    # "50b50a",
    # "100b100a",
    # "500b500a",
    # "1000b1000a",
    # "10000b10000a",
    "full",
]

pretrained_opts = [
    # False, 
    True,
]

window_opts = [
    10,
    100,
]

# Generate all combinations
for (dataset, scenario, logic_file), feature_group, window_size, subset, pretrained in product(dataset_opts, feature_group_opts, window_opts, subset_opts, pretrained_opts):

    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",

        "--dataset", str(dataset),
        "--scenario", str(scenario),
        "--logic_file", str(logic_file),
        "--feature_group", str(feature_group),
        "--subset", str(subset),
        "--window_size", str(window_size),
        "--epochs", str(30),
    ]

    if pretrained:
        cmd.append("--pretrained")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
