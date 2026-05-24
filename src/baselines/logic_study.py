import subprocess
from itertools import product


classifiers = [
    "multiclass",
    "ensemble",
]

scenarios = [
    ("darpa2000", "s1_inside"),
    # ("aitv2", "santos"),
]

feature_group = "base"
subset = "full"
window_size = 100
learning_rate = 1e-3
epochs = 50
cv_folds = 1
experiment = "logic_study"


# uv run python -m src.baselines.logic_study
for classifier, (dataset, scenario) in product(classifiers, scenarios):
    
    data_dir = f"data/processed/{dataset}/{scenario}/{feature_group}/windowed/w{window_size}"
    out_dir = f"experiments/{dataset}/{scenario}/{experiment}/baselines"

    cmd = [
        "uv", "run", "python", "-m", "src.baselines.lstm",
    
        "--classifier", str(classifier),
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),

        "--feature_group", str(feature_group),
        "--subset", str(subset),
        "--window_size", str(window_size),
        "--learning_rate", str(learning_rate),
        "--epochs", str(epochs),
        "--cv_folds", str(cv_folds),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
