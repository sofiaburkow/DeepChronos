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

    num_phases = 6  # phases 0..5
    phases = list(range(num_phases))
    num_misclassified = [counts.get(p, 0) for p in phases]

    print(f"Number of misclassified samples: {len(misclassified_indices)}")
    print("Misclassified samples per phase:")
    for phase, count in zip(phases, num_misclassified):
        print(f"Phase {phase}: {count} samples")
    
    # Plot distribution of misclassified samples across phases
    plt.figure(figsize=(10, 6))
    plt.bar(phases, num_misclassified, color='red', alpha=0.7)
    plt.xlabel('Phases')
    plt.ylabel('Num. of Misclassified Samples')
    plt.title('Misclassified Samples Across Phases')
    plt.xticks(phases)
    os.makedirs(f"experiments/results/{output_dir}", exist_ok=True)
    plt.savefig(f'experiments/results/{output_dir}/misclassified_samples.png')


if __name__ == "__main__":
    pass