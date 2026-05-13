import matplotlib.pyplot as plt
import pandas as pd


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


def temp_metrics(misclassified, f1, phase_starts):

    df = misclassified.copy()

    df["start_time_dt"] = pd.to_datetime(df["start_time_dt"], errors="coerce")
    df["end_time_dt"]   = pd.to_datetime(df["end_time_dt"],   errors="coerce")
    
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
    
    wrong = df[df['phase'] != df['y_pred']]
    causal = wrong[wrong['causal_violation']]
    regression = wrong[(~wrong['causal_violation']) & (wrong['regression_violation'])]
    plausible = wrong[(~wrong['causal_violation']) & (~wrong['regression_violation'])]
    
    total_wrong = len(wrong)
    causal_rate = len(causal) / total_wrong if total_wrong > 0 else 0
    regression_rate = len(regression) / total_wrong if total_wrong > 0 else 0
    plausible_rate = len(plausible) / total_wrong if total_wrong > 0 else 0

    temp_score = f1 - 0.5 * causal_rate - 0.2 * regression_rate
    
    temp_metrics_dict = {
        "total_wrong": total_wrong,
        "num_causal": len(causal),
        "num_regression": len(regression),
        "num_plausible": len(plausible),
        "causal_rate": causal_rate,
        "regression_rate": regression_rate,
        "plausible_rate": plausible_rate,
        "temp_score": temp_score
    }

    return wrong, causal, regression, plausible, temp_metrics_dict


def plot_temp_consistency(
    df, 
    phase_bounds, 
    causal,
    regression,
    plausible,
    temp_metrics_dict,
    exp_name, 
    out_dir,
    attack_start=None, 
    attack_end=None,
    save_plot=True, 
    show_plot=True,
):
    
    df = df.copy()

    # Trim to specified attack window, if provided
    if attack_start is not None and attack_end is not None:
        df["start_time_dt"] = pd.to_datetime(df["start_time_dt"], errors="coerce")
        df["end_time_dt"]   = pd.to_datetime(df["end_time_dt"],   errors="coerce")
        df = df[(df["start_time_dt"] >= attack_start) & (df["end_time_dt"] <= attack_end)]
        
        causal      = causal[(causal["start_time_dt"] >= attack_start) & (causal["end_time_dt"] <= attack_end)]
        regression  = regression[(regression["start_time_dt"] >= attack_start) & (regression["end_time_dt"] <= attack_end)]
        plausible   = plausible[(plausible["start_time_dt"] >= attack_start) & (plausible["end_time_dt"] <= attack_end)]

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

    ax.scatter(causal["start_time_dt"], causal["y_pred"], s=70, marker="X",
               color="red", alpha=0.9, label="Causal violations")

    ax.scatter(regression["start_time_dt"], regression["y_pred"], s=70, marker="X",
               color="orange", alpha=0.9, label="Regression violations")

    # ax.scatter(plausible["start_time_dt"], plausible["y_pred"], s=70, marker="X",
    #            color="blue", alpha=0.9, label="Temporal‑plausible errors")

    ax.set_ylim(-0.5, max(values) + 2)
    ax.set_xlim(df["start_time_dt"].min(), df["start_time_dt"].max())
    ax.set_xlabel("Time")
    ax.set_ylabel("Phase")
    ax.set_title(f"Temporal Consistency - Misclassifications")
    ax.set_yticks(sorted(df["phase"].unique()))
    ax.grid(alpha=0.2)

    textstr = (
        f"Causal: {temp_metrics_dict['num_causal']} "
        f"({temp_metrics_dict['causal_rate']:.1%})\n"
        f"Regression: {temp_metrics_dict['num_regression']} "
        f"({temp_metrics_dict['regression_rate']:.1%})\n"
        f"Plausible: {temp_metrics_dict['num_plausible']} "
        f"({temp_metrics_dict['plausible_rate']:.1%})"
    )

    ax.text(
        0.02, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    
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

