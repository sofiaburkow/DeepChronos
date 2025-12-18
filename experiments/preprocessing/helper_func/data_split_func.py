import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(df, split_size, stratify_col="phase", random_state=123):
    '''
    Perform a stratified split of the dataframe into train and test sets.
    '''
    test_size = 1 - split_size
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df[stratify_col], random_state=random_state
    )

    return df_train, df_test


def host_temporal_split(df, train_ratio=0.6):
    '''
    Performs a host-based temporal split.
    For each (src_ip, dst_ip) pair:
        - Sort by start_time
        - First train_ratio fraction -> train
        - Remaining -> test
    '''
    train_parts = []
    test_parts = []

    # Group by host pair
    grouped = df.groupby(["src_ip", "dst_ip"])

    for (src, dst), group in grouped:
        # Sort by timestamp
        group_sorted = group.sort_values("start_time")

        # Compute split index
        split_idx = int(len(group_sorted) * train_ratio)

        # Append to lists
        train_parts.append(group_sorted.iloc[:split_idx])
        test_parts.append(group_sorted.iloc[split_idx:])

    # Concatenate all host groups
    df_train = pd.concat(train_parts).reset_index(drop=True)
    df_test = pd.concat(test_parts).reset_index(drop=True)

    return df_train, df_test


def prepare_phase_dataset(df, target_phase):
    df = df.copy()

    # Binary label for target phase
    df['y'] = (df['phase'] == target_phase).astype(int)

    # Sort by start_time
    df = df.sort_values('start_time').reset_index(drop=True)

    return df


def temporal_split_with_attack(df, target_phase, test_ratio=0.4):
    '''
    Temporal split that ensures both train and test contain attacks.
    '''
    
    df_phase = prepare_phase_dataset(df, target_phase)

    split_idx = int(len(df_phase) * (1 - test_ratio))

    df_train = df_phase.iloc[:split_idx]
    df_test  = df_phase.iloc[split_idx:]

    # Check if attacks exist in both
    if df_train["y"].sum() == 0 or df_test["y"].sum() == 0:
        raise ValueError(
            "Temporal split resulted in no attacks in train or test. "
            "Adjust split or split inside attack window."
        )

    return df_train, df_test


def build_sequences(df, window_size=5):
    df = df.sort_values("start_time")

    X, y = [], []

    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]

        X.append(window.values)
        y.append(window["y"].iloc[-1])  # predict next-step state

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Command: uv run python experiments/preprocessing/helper_func/data_split_func.py

    # Example usage
    df = pd.read_csv("data/DARPA_2000/Scenario_One/inside/inside_labeled_flows_all.csv")

    # df_train_s, df_test_s = stratified_split(df, split_size=0.6)
    # print("Stratified Split:")
    # print("Train shape:", df_train_s.shape)
    # print("Test shape:", df_test_s.shape)

    # df_train_ht, df_test_ht = host_temporal_split(df, train_ratio=0.6)
    # print("\nHost-Temporal Split:")
    # print("Train shape:", df_train_ht.shape)
    # print("Test shape:", df_test_ht.shape)

    df_train_t, df_test_t = temporal_split_with_attack(df, target_phase=2, test_ratio=0.4)
    print("\nTemporal Split with Attack:")
    print("Train shape:", df_train_t.shape)
    print("Test shape:", df_test_t.shape)