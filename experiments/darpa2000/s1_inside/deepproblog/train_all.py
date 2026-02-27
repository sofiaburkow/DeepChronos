import subprocess
from itertools import product

# Config options
window_opts = [
    # 10, 
    50, 
    # 100,
    ]
functions = [
    # "ddos", 
    "multi_step",
    ]
lookbacks = [
    None, 
    # 20000,
    ]
resampled_opts = [
    False, 
    True
    ]
pretrained_opts = [
    # False, 
    True
    ]

# Generate all combinations
for func, lookback, resampled, pretrained, window in product(functions, lookbacks, resampled_opts, pretrained_opts, window_opts):
    cmd = ["uv", "run", "python", "-m", "src.deepproblog.train", "--function_name", func]

    if lookback is not None:
        cmd.extend(["--lookback_limit", str(lookback)])
    if resampled:
        cmd.append("--resampled")
    if pretrained:
        cmd.append("--pretrained")
    if window is not None:
        cmd.extend(["--window_size", str(window)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
