import argparse
from pathlib import Path
from itertools import product

import numpy as np


def sample_per_class(labels, target_per_class, seed):
    """
    Flexible class-wise sampling.

    Args:
        labels (np.ndarray): array of class labels
        target_per_class (dict): {class_label: target_count}
        seed (int): random seed

    Behavior:
        - Downsamples if class > target
        - Oversamples (with replacement) if class < target
        - Ignores classes not in target_per_class
    """
    rng = np.random.default_rng(seed)

    all_indices = np.arange(len(labels))
    sampled_indices = []

    for cls, target_n in target_per_class.items():
        cls_indices = all_indices[labels == cls]

        if len(cls_indices) == 0:
            print(f"[!] Warning: class {cls} has no samples, skipping.")
            continue

        if len(cls_indices) >= target_n:
            # Downsample
            sampled = rng.choice(cls_indices, size=target_n, replace=False)
        else:
            # Oversample
            sampled = rng.choice(cls_indices, size=target_n, replace=True)

        sampled_indices.append(sampled)

    sampled_indices = np.concatenate(sampled_indices)
    sampled_indices = rng.permutation(sampled_indices)

    return sampled_indices


def build_target_dict(labels, benign_target, attack_target):
    """
    Automatically create class targets:
    - class 0 (benign) gets larger sample
    - all attack phases get equal smaller samples
    """
    classes = np.unique(labels)

    target = {}
    for cls in classes:
        if cls == 0:
            target[cls] = benign_target
        else:
            target[cls] = attack_target

    return target


def main(data_dir, benign_target, attack_target, seed):

    print("[+] Loading training labels...")
    y_train = np.load(data_dir / "y_train.npy")

    # --- Build target distribution ---
    target_per_class = build_target_dict(
        y_train,
        benign_target=benign_target,
        attack_target=attack_target,
    )

    # --- Sample ---
    indices = sample_per_class(
        y_train,
        target_per_class=target_per_class,
        seed=seed,
    )

    print("Dataset size:", len(indices))

    class_distribution = {
        cls: int(np.sum(y_train[indices] == cls))
        for cls in np.unique(y_train)
    }

    print("Class distribution:", class_distribution)

    subset_dir = data_dir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    out_path = subset_dir / f"train_{benign_target}b{attack_target}a.npy"
    np.save(out_path, indices)
    print(f"[✓] Saved {out_path}")
    

if __name__ == "__main__":
    # uv run python -m src.feature_engineering.sample_data --dataset aitv2 --scenario santos

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="darpa2000")
    parser.add_argument("--scenario", default="s1_inside")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    feature_groups = [
        "full", 
        "reduced",
        "behavioral"
    ]

    window_sizes = [
        10, 
        100
    ]

    benign_targets = [5, 10, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000]
    attack_targets = [5, 10, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000]

    for feature_group, window_size, benign_target, attack_target in product(
        feature_groups,
        window_sizes,
        benign_targets,
        attack_targets
    ):
        data_dir = Path(
            f"data/processed/{args.dataset}/{args.scenario}/{feature_group}/windowed/w{window_size}"
        )

        print(f"\nCreating subsets for feature group {feature_group} and window size {window_size}")

        main(
            data_dir=data_dir,
            benign_target=benign_target,
            attack_target=attack_target,
            seed=args.seed,
        )