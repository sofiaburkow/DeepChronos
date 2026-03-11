import matplotlib.pyplot as plt


def is_hard_violation(row, phase_start):
    pred_phase = row['y_pred']
    t = row['t_rel']

    if pred_phase == 0:
        return False

    for prev_p in range(1, pred_phase):
        if prev_p in phase_start:
            if t < phase_start[prev_p]:
                return True

    return False


def is_soft_violation(row, phase_start):
    pred_phase = row['y_pred']
    t = row['t_rel']

    if pred_phase == 0:
        return False

    next_phase = pred_phase + 1

    if next_phase in phase_start:
        if t >= phase_start[next_phase]:
            return True

    return False


def mis_metrics(mis_df, phase_start):
    
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

    wrong = mis_df[mis_df['phase'] != mis_df['y_pred']]

    hard = wrong[wrong['hard_violation']]
    soft = wrong[(~wrong['hard_violation']) & (wrong['soft_violation'])]
    plausible = wrong[(~wrong['hard_violation']) & (~wrong['soft_violation'])]

    return wrong, hard, soft, plausible


def plot_mis_predictions(df, phase_bounds, plausible, soft, hard, total_wrong, soft_rate, hard_rate, exp_name, out_dir):
    
    plt.figure(figsize=(14, 5))

    # --- True phase over time ---
    plt.plot(
        df['t_rel'],
        df['phase'],
        color='black',
        linewidth=1,
        alpha=0.3,
        label="True phase"
    )

    # --- Shaded phase intervals ---
    for phase, bounds in phase_bounds.iterrows():
        plt.axvspan(
            bounds['min'],
            bounds['max'],
            alpha=0.05
        )

    # --- Plausible errors ---
    plt.scatter(
        plausible['t_rel'],
        plausible['y_pred'],
        s=60,
        alpha=0.8,
        label='Wrong but temporally plausible'
    )

    # --- Soft violations ---
    plt.scatter(
        soft['t_rel'],
        soft['y_pred'],
        s=70,
        marker='s',
        alpha=0.9,
        label='Soft violation (phase regression)'
    )

    # --- Hard violations ---
    plt.scatter(
        hard['t_rel'],
        hard['y_pred'],
        s=90,
        marker='X',
        edgecolor='black',
        linewidth=0.6,
        zorder=3,
        label='Hard violation (causal impossibility)'
    )

    # --- Cosmetics ---
    plt.xlabel("Relative time")
    plt.ylabel("Phase")
    plt.title(f"Temporal Consistency: {exp_name}")

    plt.yticks(sorted(df['phase'].unique()))
    plt.grid(alpha=0.2)

    plt.text(
        0.02,
        0.95,
        f"Total wrong: {total_wrong}\n"
        f"Hard: {len(hard)} ({hard_rate:.2%})\n"
        f"Soft: {len(soft)} ({soft_rate:.2%})",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )

    plt.legend(frameon=False)
    plt.tight_layout()

    # Save to file
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp_name}_temp_plot.png"
    plt.savefig(out_path)

    plt.show()