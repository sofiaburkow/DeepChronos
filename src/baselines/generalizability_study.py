import subprocess
from itertools import product

classifiers = [
    "multiclass",
]

scenarios = [
    ("aitv2", "santos_fox"),
    ("aitv2", "fox"),
    # ("darpa2000", "s1_inside_s1_dmz"),
    # ("darpa2000", "s1_dmz"),
]

feature_group = "base"
subset = "full"
# window_size = 100
window_size = 10
learning_rate = 1e-3
epochs = 50
cv_folds = 1
experiment = "generalizability_study"


# uv run python -m src.baselines.generalizability_study
for classifier, (dataset, scenario) in product(classifiers, scenarios):
    
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
