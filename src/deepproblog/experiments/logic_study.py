import subprocess
from itertools import product

# dataset, scenario, logic_file
logic_opts = [
    # ("aitv2", "santos", "ait_temp_context"),
    # ("aitv2", "santos", "ait_temp"),
    ("aitv2", "santos", "ait_temp_context_baseline"),
    ("aitv2", "santos", "ait_temp_baseline"),
]

pretrained_opts = [
    False, 
    # True,
]

feature_group = "base"
subset = "full"
window_size = 100
learning_rate = 1e-3
epochs = 50
experiment = "logic_study"

# uv run python -m src.deepproblog.experiments.logic_study
for (dataset, scenario, logic_file), pretrained in product(logic_opts, pretrained_opts):

    data_dir = f"data/processed/{dataset}/{scenario}/{feature_group}/windowed/w{window_size}"
    experiment_dir = f"experiments/{dataset}/{scenario}/{experiment}/deepproblog"
    pretrained_dir = f"experiments/{dataset}/{scenario}/deepproblog/pretrained_nets/{feature_group}/w{window_size}/full/models"

    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",

        "--data_dir", str(data_dir),
        "--experiment_dir", str(experiment_dir),
        "--pretrained_dir", str(pretrained_dir),

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
