import pandas as pd


def get_phase_bounds(dataset, test_scenario):
    df = pd.read_csv(
        f"../../data/interim/{dataset}/{test_scenario}/flows_labeled/all_flows_behavioral.csv"
    )
    df = df.sort_values("start_time").reset_index(drop=True)

    df["start_time_dt"] = pd.to_datetime(df["start_time_dt"], errors="coerce")

    phase_bounds = (
        df[df["phase"] != "benign"]
        .groupby('phase')['start_time_dt']
        .agg(['min', 'max'])
    )

    return phase_bounds


def is_causal_violation(row, phase_starts):
    pred_phase = row['y_pred']
    t = row['start_time_dt']

    if pred_phase == 0 or pred_phase == 1:
        return False
    
    prev_phase = pred_phase - 1
    
    if prev_phase in phase_starts:
        if t < phase_starts[prev_phase]:
            return True

    return False


def is_regression_violation(row, phase_starts):
    pred_phase = row['y_pred']
    t = row['start_time_dt']

    if pred_phase == 0:
        return False

    next_phase = pred_phase + 1

    if next_phase in phase_starts:
        if t >= phase_starts[next_phase]:
            return True

    return False


def temp_metrics(df, f1, dataset, test_scenario):
    df = df.copy()

    phase_bounds = get_phase_bounds(dataset, test_scenario)
    phase_starts = phase_bounds['min'].to_dict()

    df["start_time_dt"] = pd.to_datetime(df["start_time_dt"], errors="coerce")
    df["end_time_dt"] = pd.to_datetime(df["end_time_dt"], errors="coerce")

    df['causal_violation'] = df.apply(
        is_causal_violation,
        axis=1,
        phase_starts=phase_starts
    )

    df['regression_violation'] = df.apply(
        is_regression_violation,
        axis=1,
        phase_starts=phase_starts
    )
    
    correct_df = df[df['phase'] == df['y_pred']]
    wrong_df = df[df['phase'] != df['y_pred']]
    causal_df = wrong_df[wrong_df['causal_violation']]
    regression_df = wrong_df[(~wrong_df['causal_violation']) & (wrong_df['regression_violation'])]

    num_samples = len(df)
    num_wrong = len(wrong_df)
    num_causal = len(causal_df)
    num_regression = len(regression_df)

    error_rate = num_wrong / num_samples if num_samples > 0 else 0  
    causal_rate = num_causal / num_wrong if num_wrong > 0 else 0
    regression_rate = num_regression / num_wrong if num_wrong > 0 else 0

    temp_score = f1 - 0.5 * causal_rate - 0.2 * regression_rate

    dataframes = {
        "correct": correct_df,
        "causal": causal_df,
        "regression": regression_df,
    }

    temp_metrics_dict = {
        "num_wrong": num_wrong,
        "num_causal": num_causal,
        "num_regression": num_regression,
        "error_rate": error_rate,
        "causal_rate": causal_rate,
        "regression_rate": regression_rate,
        "temp_score": temp_score
    }
    
    return dataframes, temp_metrics_dict
