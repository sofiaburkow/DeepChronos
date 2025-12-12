import os
import matplotlib.pyplot as plt
from collections import Counter

def plot_misclassified_samples(y_test, y_pred, y_phase_test, output_dir):
    '''
    Plot the distribution of misclassified samples across different phases.
    '''
    # Identify misclassified samples and count them per phase
    misclassified_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    misclassified_phases = [y_phase_test[i] for i in misclassified_indices]
    counts = Counter(misclassified_phases)

    num_phases = 6  # phases 0 = benign, 1-5 = attack phases
    phases = list(range(num_phases))
    num_misclassified = [int(counts.get(p, 0)) for p in phases]

    print(f"Number of misclassified samples: {len(misclassified_indices)}")
    print("Misclassified samples per phase:")
    for phase, count in zip(phases, num_misclassified):
        print(f"Phase {phase}: {count} samples")
    print()
    
    # Plot distribution of misclassified samples across phases
    plt.figure(figsize=(10, 6))
    plt.bar(phases, num_misclassified, color='red', alpha=0.7)
    plt.xlabel('Phases')
    plt.ylabel('Number of Misclassified Samples')
    plt.title('Misclassified Samples')
    plt.xticks(phases)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # put total in the top-right corner of the axes (axes fraction coordinates)
    total_mis = sum(num_misclassified)
    plt.text(
        0.98,
        0.98,
        f"Total misclassified: {total_mis}",
        transform=plt.gca().transAxes,
        ha='right',
        va='top',
        fontsize=14,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    os.makedirs(f"experiments/results/{output_dir}", exist_ok=True)
    plt.savefig(f'experiments/results/{output_dir}/misclassified_samples.png')
    print(f"Misclassified samples plot saved to: experiments/results/{output_dir}/misclassified_samples.png")


if __name__ == "__main__":
    pass