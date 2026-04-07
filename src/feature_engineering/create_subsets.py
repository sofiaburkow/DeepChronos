from pathlib import Path
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


def detect_majority_classes(labels, threshold_fraction=0.5):
    """
    Automatically detect majority classes based on a frequency threshold.

    Args:
        labels (np.array): Array of class labels.
        threshold_fraction (float): Classes with more than this fraction of total samples 
                                    are considered majority classes.
    
    Returns:
        list: List of majority class labels
    """
    attack_labels = labels[labels != 0]  # Exclude normal class (0)
    unique, counts = np.unique(attack_labels, return_counts=True)
    total = len(attack_labels)
    majority_classes = [cls for cls, cnt in zip(unique, counts) if cnt / total > threshold_fraction]

    return majority_classes


def downsample_majority_preserve_ratios(labels, majority_classes, fraction, seed):
    """
    Downsamples only the majority classes, preserving their internal ratios.
    Minority classes are kept fully intact.
    
    Args:
        labels (np.array): Array of class labels.
        majority_classes (list[int]): Classes to downsample.
        fraction (float): Fraction to keep of the majority classes.
        seed (int): Random seed for reproducibility.
        
    Returns:
        np.array: Sorted array of indices representing the new subset.
    """
    np.random.seed(seed)
    all_indices = np.arange(len(labels))

    # Separate majority and minority indices
    majority_idx = np.isin(labels, majority_classes)
    minority_idx = ~majority_idx

    majority_indices = all_indices[majority_idx]
    minority_indices = all_indices[minority_idx]

    # Find indices for each majority class
    downsampled_majority = []
    for cls in majority_classes:
        cls_indices = majority_indices[labels[majority_indices] == cls]
        n_keep = int(len(cls_indices) * fraction)
        sampled = np.random.choice(cls_indices, size=n_keep, replace=False)
        downsampled_majority.append(sampled)

    # Combine all majority samples + all minority samples
    final_indices = np.concatenate(downsampled_majority + [minority_indices])
    return np.sort(final_indices)


def main(data_dir, fractions, seed):

    subset_dir = data_dir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Loading training labels...")
    y_train = np.load(data_dir / "y_train.npy")

    # Automatically detect majority classes
    majority_classes = detect_majority_classes(y_train, threshold_fraction=0.5)
    majority_classes.append(0)  # Include normal class
    print(f"[+] Detected majority classes: {majority_classes}")

    for frac in fractions:
        name = f"train_{int(frac*100)}.npy"
        print(f"[+] Creating subset {frac:.2f}")

        indices = downsample_majority_preserve_ratios(
            y_train,
            majority_classes=majority_classes,
            fraction=frac,
            seed=seed,
        )

        print("Dataset size:", len(indices))
        class_distribution = {cls: np.sum(y_train[indices] == cls) for cls in np.unique(y_train)}
        print("Class distribution:", class_distribution)

        np.save(subset_dir / name, indices)
        print(f"    Saved {name} ({len(indices)} samples)")

    print("\n[✓] Subsets created successfully.")


if __name__ == "__main__":
    # uv run python -m src.feature_engineering.create_subsets --dataset aitv2 --scenario fox --feature_group all --window_size 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="darpa2000")
    parser.add_argument("--scenario", default="s1_inside")
    parser.add_argument("--feature_group", default="all")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data_dir = Path(
        f"data/processed/{args.dataset}/{args.scenario}/{args.feature_group}/windowed/w{args.window_size}"
    )

    fractions = [1.0, 0.5, 0.25, 0.10, 0.05]

    main(
        data_dir,
        fractions,
        args.seed,
    )