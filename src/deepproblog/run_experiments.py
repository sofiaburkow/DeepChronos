import subprocess
from itertools import product

dataset_scenario_opts = [
    ("darpa2000", "s1_inside"),
    ("darpa2000", "s1_dmz"),
    ("aitv2", "santos"),
    ("aitv2", "fox"),
]

logic_opts = [
    ("darpa", 1),
    ("ait", 1),
]

feature_group_opts = [
    "full",
    "reduced",
    "aug",
]

subset_opts = [
    "5b5a",
    "10b10a",
    "20b20a",
    "30b30a",
    "50b50a",
    "100b100a",
    "500b500a",
    "1000b1000a",
    "full"
]

pretrained_opts = [
    False, 
    True
]

window_opts = [
    10,
    100,
]

# Generate all combinations
for (dataset, scenario), feature_group, (logic_file, num_networks), window_size, subset, pretrained in product(dataset_scenario_opts, feature_group_opts, logic_opts, window_opts, subset_opts, pretrained_opts):
    
    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",

        "--dataset", str(dataset),
        "--scenario", str(scenario),
        "--feature_group", str(feature_group),
        "--logic_file", str(logic_file),
        "--num_networks", str(num_networks),
        "--subset", str(subset),
        "--window_size", str(window_size),
        "--epochs", 20,
    ]

    if pretrained:
        cmd.append("--pretrained")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
