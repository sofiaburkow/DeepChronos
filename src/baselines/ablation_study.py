import subprocess
from itertools import product


classifiers = [
    # "multiclass",
    "ensemble",
]

scenarios = [
    ("darpa2000", "s1_inside"),
    # ("aitv2", "santos"),
]

feature_groups = [
    # "base",
    "flowonly", 
    # "portaware"
]

window_sizes = [
    10,
    100,
]

learning_rates = [
    1e-3,
    1e-4,
]

experiment_name = "ablation_study"

# uv run python -m src.baselines.ablation_study
for classifier, (dataset, scenario), feature_group, window_size, learning_rate in product(classifiers, scenarios, feature_groups, window_sizes, learning_rates):
    
    data_dir = f"data/processed/{dataset}/{scenario}/{feature_group}/windowed/w{window_size}"
    out_dir = f"experiments/{dataset}/{scenario}/{experiment_name}/{classifier}"

    cmd = [
        "uv", "run", "python", "-m", "src.baselines.lstm",
    
        "--classifier", str(classifier),
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),

        "--feature_group", str(feature_group),
        "--window_size", str(window_size),
        "--learning_rate", str(learning_rate),
        
        "--subset", "full",
        "--epochs", str(50),
        "--cv_folds", str(5),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
