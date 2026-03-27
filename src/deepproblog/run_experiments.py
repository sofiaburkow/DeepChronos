import subprocess
from itertools import product

# Config options

logic_opts = [
    # ("darpa", 5),
    ("darpa_neg", 5),
    # ("darpa_one_net", 1),
    # ("darpa_flags", 5),
    # ("darpa_flags_neg", 5),
    # ("darpa_flags_one_net", 1),
    ]

dataset_variant_opts = [
    "original",
    "up", 
    "down",
    ]

pretrained_opts = [
    False, 
    True
    ]

window_opts = [
    10, 
    # 50, 
    # 100,
    ]

# Generate all combinations
for (logic_file, num_networks), window, dataset_variant, pretrained in product(logic_opts, window_opts, dataset_variant_opts, pretrained_opts):
    
    cmd = [
        "uv", "run", "python", "-m", "src.deepproblog.train",
        "--logic_file", str(logic_file),
        "--num_networks", str(num_networks),
        "--window_size", str(window),
        "--dataset_variant", str(dataset_variant)
    ]

    if pretrained:
        cmd.append("--pretrained")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
