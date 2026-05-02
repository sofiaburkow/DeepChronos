import matplotlib.pyplot as plt
import pandas as pd


def is_hard_violation(row, phase_start):
    pred_phase = row['y_pred']
    t = row['start_time_dt']

    if pred_phase == 0 or pred_phase == 1:
        return False
    
    prev_phase = pred_phase - 1
    
    if prev_phase in phase_start:
        if t < phase_start[prev_phase]:
            return True

    return False


def is_soft_violation(row, phase_start):
    pred_phase = row['y_pred']
    t = row['start_time_dt']

    if pred_phase == 0:
        return False

    next_phase = pred_phase + 1

    if next_phase in phase_start:
        if t >= phase_start[next_phase]:
            return True

    return False


def mis_metrics(mis_df, phase_start):

    mis_df["start_time_dt"] = pd.to_datetime(mis_df["start_time_dt"], errors="coerce")
    mis_df["end_time_dt"]   = pd.to_datetime(mis_df["end_time_dt"],   errors="coerce")
    
    mis_df['hard_violation'] = mis_df.apply(
        is_hard_violation,
        axis=1,
        phase_start=phase_start
    )

    mis_df['soft_violation'] = mis_df.apply(
        is_soft_violation,
        axis=1,
        phase_start=phase_start
    )

    wrong_df = mis_df[mis_df['phase'] != mis_df['y_pred']]

    hard_df = wrong_df[wrong_df['hard_violation']]
    soft_df = wrong_df[(~wrong_df['hard_violation']) & (wrong_df['soft_violation'])]
    plausible_df = wrong_df[(~wrong_df['hard_violation']) & (~wrong_df['soft_violation'])]

    return wrong_df, hard_df, soft_df, plausible_df


def temp_metrics(f1, wrong, hard, soft):

    total_wrong = len(wrong)
    hard_rate = len(hard) / total_wrong if total_wrong > 0 else 0
    soft_rate = len(soft) / total_wrong if total_wrong > 0 else 0

    temp_score = f1 - 0.5 * hard_rate - 0.2 * soft_rate

    return hard_rate, soft_rate, temp_score 


def plot_mis_predictions(
    df, phase_bounds, 
    plausible, soft, hard, 
    exp_name, out_dir,
    save_plot=True, show_plot=True,
    attack_start=None, attack_end=None,
):
    df = df.copy()
    # --- Optionally trim DF ---
    if attack_start is not None and attack_end is not None:
        df["start_time_dt"] = pd.to_datetime(df["start_time_dt"], errors="coerce")
        df["end_time_dt"]   = pd.to_datetime(df["end_time_dt"],   errors="coerce")
        df = df[(df["start_time_dt"] >= attack_start) & (df["end_time_dt"] <= attack_end)]
        
        plausible = plausible[(plausible["start_time_dt"] >= attack_start) & (plausible["end_time_dt"] <= attack_end)]
        soft      = soft[(soft["start_time_dt"] >= attack_start) & (soft["end_time_dt"] <= attack_end)]
        hard      = hard[(hard["start_time_dt"] >= attack_start) & (hard["end_time_dt"] <= attack_end)]

    fig, ax = plt.subplots(figsize=(14, 5))

    edges  = phase_bounds["min"].tolist() + [phase_bounds["max"].iloc[-1]]
    values = phase_bounds.index.tolist()

    ax.stairs(
        values=values,
        edges=edges,
        linewidth=1,
        color="black",
        label="True phase"
    )

    values.append(5)
    ax.fill_between(
        edges,                   
        values,                  
        [min(values) - 0.5]*len(values),
        step="post",
        color="lightgray",
        alpha=0.2
    )

    ax.scatter(plausible["start_time_dt"], plausible["y_pred"], s=70, marker="X",
               color="purple", alpha=0.9, label="Temporal‑plausible errors")

    ax.scatter(soft["start_time_dt"], soft["y_pred"], s=70, marker="X",
               color="orange", alpha=0.9, label="Regression violations")

    ax.scatter(hard["start_time_dt"], hard["y_pred"], s=70, marker="X",
               color="red", alpha=0.9, label="Causal violations")


    ax.set_ylim(-0.5, max(values) + 2)
    ax.set_xlim(df["start_time_dt"].min(), df["start_time_dt"].max())

    ax.set_xlabel("Relative time")
    ax.set_ylabel("Phase")
    ax.set_title(f"Temporal Consistency - Prediction Violations")
    ax.set_yticks(sorted(df["phase"].unique()))
    ax.grid(alpha=0.2)

    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        loc="upper right"
    )

    fig.tight_layout()

    if save_plot:
        out_path = out_dir / f"{exp_name}.png"
        print(f"Saving plot to {out_path}...")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
