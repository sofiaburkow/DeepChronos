import subprocess
from itertools import product


logic_opts = [
    ("darpa_flags_neg_one_net", 1),
    ("darpa_flags_neg", 5),
    ("darpa_flags_one_net", 1),
    ("darpa_neg_one_net", 1),
    ("darpa_neg", 5),
    ("darpa_one_net", 1),
]

subset_opts = [
    "500b5a",
    # "500b10a",
    # "500b20a",
    # "500b30a",
]

pretrained_opts = [
    False, 
    # True
]

window_opts = [
    10, 
    # 50, 
    # 100,
]

# Generate all combinations
for (logic_file, num_networks), window_size, subset, pretrained in product(logic_opts, window_opts, subset_opts, pretrained_opts):
    
    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",
        "--logic_file", str(logic_file),
        "--num_networks", str(num_networks),
        "--window_size", str(window_size),
        "--subset", str(subset)
    ]

    if pretrained:
        cmd.append("--pretrained")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
