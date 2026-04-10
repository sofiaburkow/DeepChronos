from pathlib import Path
import argparse
import numpy as np
from itertools import product


# ============================================================
# CLASS DETECTION
# ============================================================

def detect_majority_classes(labels, threshold_fraction=0.3):
    """
    Detect majority attack classes (excluding normal=0).
    """
    attack_labels = labels[labels != 0]

    unique, counts = np.unique(attack_labels, return_counts=True)
    total = len(attack_labels)

    return [
        cls for cls, cnt in zip(unique, counts)
        if cnt / total > threshold_fraction
    ]


def detect_minority_classes(labels, threshold_fraction=0.05):
    """
    Detect minority classes.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    return [
        cls for cls, cnt in zip(unique, counts)
        if cnt / total < threshold_fraction
    ]


# ============================================================
# DOWNSAMPLING
# ============================================================

def downsample_majority_preserve_ratios(
    labels,
    majority_classes,
    fraction,
    rng,
):
    """
    Downsample only majority classes while preserving ratios.
    Minority classes remain untouched.
    """
    all_indices = np.arange(len(labels))

    majority_mask = np.isin(labels, majority_classes)
    minority_mask = ~majority_mask

    majority_indices = all_indices[majority_mask]
    minority_indices = all_indices[minority_mask]

    downsampled_majority = []

    for cls in majority_classes:
        cls_indices = majority_indices[labels[majority_indices] == cls]

        if len(cls_indices) == 0:
            continue

        n_keep = max(1, int(len(cls_indices) * fraction))

        sampled = rng.choice(cls_indices, size=n_keep, replace=False)
        downsampled_majority.append(sampled)

    final_indices = np.concatenate(
        downsampled_majority + [minority_indices]
    )

    return final_indices


# ============================================================
# OVERSAMPLING
# ============================================================

def oversample_minority_preserve_ratios(
    labels,
    base_indices,
    minority_classes,
    target_size_per_class,
    rng,
):
    """
    Oversample minority classes via index duplication.
    """

    extra_indices = []

    for cls in minority_classes:

        cls_idx = base_indices[labels[base_indices] == cls]

        if len(cls_idx) == 0:
            continue

        n_needed = target_size_per_class - len(cls_idx)

        if n_needed > 0:
            sampled = rng.choice(cls_idx, size=n_needed, replace=True)
            extra_indices.append(sampled)

    if extra_indices:
        base_indices = np.concatenate([base_indices] + extra_indices)

    return base_indices


# ============================================================
# MASTER BALANCING PIPELINE
# ============================================================

def balance_indices(labels, majority_classes, fraction, seed):
    """
    Hybrid balancing:
      1. Downsample majority
      2. Detect minorities
      3. Oversample minorities
      4. Shuffle indices
    """

    rng = np.random.default_rng(seed)

    # ---- Downsample large classes
    indices = downsample_majority_preserve_ratios(
        labels,
        majority_classes,
        fraction,
        rng,
    )

    # ---- Detect minorities AFTER downsampling
    minority_classes = detect_minority_classes(labels[indices], 0.05)

    # ---- Compute robust target size
    unique, counts = np.unique(labels[indices], return_counts=True)
    target_size = int(np.median(counts))

    # ---- Oversample minorities
    indices = oversample_minority_preserve_ratios(
        labels,
        indices,
        minority_classes,
        target_size,
        rng,
    )

    # ---- IMPORTANT: shuffle duplicated samples
    indices = rng.permutation(indices)

    return indices


# ============================================================
# MAIN
# ============================================================

def main(data_dir, fractions, seed):

    subset_dir = data_dir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Loading training labels...")
    y_train = np.load(data_dir / "y_train.npy")

    # Detect majority classes automatically
    majority_classes = detect_majority_classes(
        y_train,
        threshold_fraction=0.5,
    )

    majority_classes.append(0)  # include normal traffic

    print(f"[+] Detected majority classes: {majority_classes}")

    for frac in fractions:

        name = f"train_{int(frac*100)}.npy"
        print(f"\n[+] Creating subset {frac:.2f}")

        indices = balance_indices(
            y_train,
            majority_classes,
            fraction=frac,
            seed=seed,
        )

        print("Dataset size:", len(indices))

        class_distribution = {
            cls: int(np.sum(y_train[indices] == cls))
            for cls in np.unique(y_train)
        }

        print("Class distribution:", class_distribution)

        np.save(subset_dir / name, indices)
        print(f"    Saved {name}")


    print("\n[✓] Subsets created successfully.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    # uv run python -m src.feature_engineering.create_subsets --dataset aitv2 --scenario fox
    # uv run python -m src.feature_engineering.create_subsets --dataset darpa2000 --scenario s1_inside_dmz

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="darpa2000")
    parser.add_argument("--scenario", default="s1_inside")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    feature_groups = ["all", "sub"]
    window_sizes = [10, 50, 100]
    fractions = [1.0, 0.5, 0.25, 0.10, 0.05]

    for feature_group, window_size, frac in product(
        feature_groups,
        window_sizes,
        fractions,
    ):

        data_dir = Path(
            f"data/processed/{args.dataset}/{args.scenario}/{feature_group}/windowed/w{window_size}"
        )

        main(
            data_dir,
            [frac],
            args.seed,
        )