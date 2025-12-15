import pandas as pd
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


if __name__ == "__main__":
    # Command: uv run python experiments/preprocessing/helper_func/data_split_func.py

    # Example usage
    df = pd.read_csv("data/DARPA_2000/Scenario_One/inside/inside_labeled_flows_all.csv")

    df_train_s, df_test_s = stratified_split(df, split_size=0.6)
    print("Stratified Split:")
    print("Train shape:", df_train_s.shape)
    print("Test shape:", df_test_s.shape)

    df_train_ht, df_test_ht = host_temporal_split(df, train_ratio=0.6)
    print("\nHost-Temporal Split:")
    print("Train shape:", df_train_ht.shape)
    print("Test shape:", df_test_ht.shape)