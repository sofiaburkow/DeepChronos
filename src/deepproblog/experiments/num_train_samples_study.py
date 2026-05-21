import subprocess
from itertools import product

# dataset, scenario, logic_file
logic_opts = [
    # ("darpa2000", "s1_inside", "darpa_logic"),
    ("aitv2", "santos", "ait_temp_context"),
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
    "5000b5000a",
    "10000b10000a",
]

feature_group = "base"
window_size = 100
learning_rate = 1e-3
epochs = 50

for (dataset, scenario, logic_file), subset in product(logic_opts, subset_opts):

    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",

        "--dataset", str(dataset),
        "--scenario", str(scenario),
        "--logic_file", str(logic_file),
        "--subset", str(subset),
        
        "--feature_group", str(feature_group),
        "--window_size", str(window_size),
        "--learning_rate", str(learning_rate),
        "--epochs", str(epochs),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
