import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def stratified_split(df, train_ratio, stratify_col, random_state):
    '''
    Perform a stratified split of the dataframe into train and test sets.
    '''
    test_size = 1 - train_ratio
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df[stratify_col], random_state=random_state
    )

    return df_train, df_test


def host_temporal_split(df, train_ratio):
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


def prepare_phase_dataset(df, target_phase, label_name='y', sort_by_time=False):
    '''
    Prepare dataset for a specific target phase:
    - Create binary label 'y' where 1 indicates the target phase, else 0
    - Optionally sort by start_time
    '''
    df = df.copy()

    # Binary label for target phase
    df[label_name] = (df['phase'] == target_phase).astype(int)

    if sort_by_time:
        # Sort by start_time
        df = df.sort_values("start_time").reset_index(drop=True)

    return df


def build_sequences(X, y, y_phase, window_size):
    X_sequences, y_sequences, y_phase_sequences = [], [], []
    for i in range(X.shape[0] - window_size + 1):
        X_window = X[i:i+window_size]
        y_window = y[i:i+window_size]
        y_phase_window = y_phase[i:i+window_size]
        X_sequences.append(X_window.toarray())  # convert sparse to dense
        y_sequences.append(y_window.iloc[-1])  # predict next-step state
        y_phase_sequences.append(y_phase_window.iloc[-1])  # predict next-step phase

    return np.array(X_sequences), np.array(y_sequences), np.array(y_phase_sequences)


def temporal_split(X, y, y_phase, train_ratio, window_size, random_state):
    '''
    Temporal split that ensures both train and test contain attacks.
    '''
    X, y, y_phase = shuffle(X, y, y_phase, random_state=random_state)

    split_idx = int(len(X) * train_ratio) - window_size + 1

    # Train data
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    y_phase_train = y_phase[:split_idx]
    # Test data
    X_test  = X[split_idx:]
    y_test  = y[split_idx:]
    y_phase_test = y_phase[split_idx:]

    # Check if attacks exist in both
    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ValueError(
            "Temporal split resulted in no attacks in train or test. "
            "Adjust split or split inside attack window."
        )

    return X_train, X_test, y_train, y_test, y_phase_train, y_phase_test


if __name__ == "__main__":
    # Command: uv run python experiments/preprocessing/helper_func/data_split_func.py

    seed = 123

    # Example usage
    df = pd.read_csv("data/DARPA_2000/Scenario_One/inside/inside_labeled_flows_all.csv")

    df_train_s, df_test_s = stratified_split(
        df=df, 
        train_ratio=0.6, 
        stratify_col="phase", 
        random_state=seed
    )
    print("Stratified Split:")
    print("Train shape:", df_train_s.shape)
    print("Test shape:", df_test_s.shape)

    df_train_ht, df_test_ht = host_temporal_split(
        df=df, 
        train_ratio=0.6
    )
    print("\nHost-Temporal Split:")
    print("Train shape:", df_train_ht.shape)
    print("Test shape:", df_test_ht.shape)
    
    