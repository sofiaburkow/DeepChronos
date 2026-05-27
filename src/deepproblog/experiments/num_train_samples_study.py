import subprocess
from itertools import product

# dataset, scenario, logic_file, window_size
logic_opts = [
    # ("darpa2000", "s1_inside", "darpa_temp_context", 10),
    ("darpa2000", "s1_inside", "darpa_temp", 10),
    # ("aitv2", "santos", "ait_temp_context", 100),
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
pretrained = False
learning_rate = 1e-3
epochs = 50
experiment = "num_train_samples_study"

# uv run python -m src.deepproblog.experiments.num_train_samples_study
for (dataset, scenario, logic_file, window_size), subset in product(logic_opts, subset_opts):

    data_dir = f"data/processed/{dataset}/{scenario}/{feature_group}/windowed/w{window_size}"
    experiment_dir = f"experiments/{dataset}/{scenario}/{experiment}/deepproblog"
    pretrained_dir = f"experiments/{dataset}/{scenario}/deepproblog/pretrained_nets/{feature_group}/w{window_size}/full/models"

    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",

        "--data_dir", str(data_dir),
        "--experiment_dir", str(experiment_dir),
        "--pretrained_dir", str(pretrained_dir),

        "--logic_file", str(logic_file),
        "--subset", str(subset),
        "--feature_group", str(feature_group),
        "--window_size", str(window_size),
        "--learning_rate", str(learning_rate),
        "--epochs", str(epochs),
    ]

    if pretrained:
        cmd.append("--pretrained")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
