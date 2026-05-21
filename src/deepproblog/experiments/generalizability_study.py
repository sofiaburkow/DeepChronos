import subprocess
from itertools import product

# dataset, scenario, logic_file
logic_opts = [
    # ("darpa2000", "s1_inside", "darpa_logic"),
    # ("darpa2000", "s1_inside", "darpa_logic_baseline"),
    # ("darpa2000", "s1_inside", "darpa_flags"),
    # ("darpa2000", "s1_inside", "darpa_flags_baseline"),

    ("aitv2", "santos", "ait_logic"),
    ("aitv2", "fox", "ait_logic"),
    ("aitv2", "santos_fox", "ait_logic"),
    ("aitv2", "santos", "ait_logic_baseline"),
    ("aitv2", "santos", "ait_flags"),
    ("aitv2", "santos", "ait_flags_baseline"),
]

pretrained_opts = [
    False, 
    True,
]

feature_group = "base"
subset = "full"
window_size = 100
learning_rate = 1e-3
epochs = 50

for (dataset, scenario, logic_file), pretrained in product(logic_opts, pretrained_opts):

    input_dir =""
    output_dir = ""

    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",

        "--dataset", str(dataset),
        "--scenario", str(scenario),
        "--logic_file", str(logic_file),

        "--feature_group", str(feature_group),
        "--subset", str(subset),
        "--window_size", str(window_size),
        "--learning_rate", str(learning_rate),
        "--epochs", str(epochs),
    ]

    if pretrained:
        cmd.append("--pretrained")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
