import subprocess
from itertools import product

# TODO: decide upon which one
classifiers = [
    "multiclass",
    # "ensemble",
]

scenarios = [
    # ("darpa2000", "s1_inside"),
    ("aitv2", "santos"),
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
cv_folds = 1
experiment = "num_train_samples_study"


# uv run python -m src.baselines.num_train_samples_study
for classifier, (dataset, scenario), subset in product(classifiers, scenarios, subset_opts):
    
    data_dir = f"data/processed/{dataset}/{scenario}/{feature_group}/windowed/w{window_size}"
    out_dir = f"experiments/{dataset}/{scenario}/{experiment}/baselines"

    cmd = [
        "uv", "run", "python", "-m", "src.baselines.lstm",
    
        "--classifier", str(classifier),
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--subset", str(subset),

        "--window_size", str(window_size),
        "--learning_rate", str(learning_rate),
        "--epochs", str(epochs),
        "--cv_folds", str(cv_folds),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
