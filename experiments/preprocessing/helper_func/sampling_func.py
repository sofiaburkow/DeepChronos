import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def check_class_distribution(y, phases=[1,2,3,4,5], plot=False):
    '''
    This function checks the class distribution of the given labels y.
    '''
    y_filtered = [label for label in y if label in phases] 
    counts = Counter(y_filtered)
    num_samples_per_phase = dict(counts)
    
    total_num_samples = sum(counts.values())
    print(f"Total number of samples: {total_num_samples}")

    for phase in phases:
        count = counts.get(phase, 0)
        percentage = (count / total_num_samples) * 100
        print(f"Phase {phase}: {count} samples ({percentage:.2f}%)")
    
    if plot:
        plt.bar(num_samples_per_phase.keys(), num_samples_per_phase.values())
        plt.xlabel('Phase')
        plt.ylabel('Number of Flows')
        plt.title('Number of Flows per Phase in Scenario One Inside Dataset')
        plt.show()

    return num_samples_per_phase


def sample_classes_random(mode, X, desired_target, label_name, phases, random_state):
    y = X[label_name] 
    counts = Counter(y)
    print('Before sampling:', Counter(counts))

    if mode == 'upsample':
        sampling_strategy = {p: desired_target for p in phases if counts.get(p, 0) < desired_target}
        print('Sampling strategy (per-phase target):', sampling_strategy)
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    elif mode == 'downsample': 
        sampling_strategy = {p: desired_target for p in phases if counts.get(p, 0) > desired_target}
        print('Sampling strategy (per-phase target):', sampling_strategy)
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    else:
        raise ValueError("Mode should be either 'upsample' or 'downsample'.")
    
    print('After sampling:', Counter(y_resampled))

    return X_resampled, y_resampled


def sample_per_phase(mode, X_per_phase, desired_target, phase, label_name, random_state):
    '''
    For now, only random over/undersampling.
    '''
    labels = X_per_phase[label_name]
    counts = Counter(labels)

    if mode == "auto":
        if counts[1] < desired_target:
            sampling_mode = 'upsample'
        else:
            sampling_mode = 'downsample'
    else:
        raise ValueError(f"Mode {mode} is not a valid per-phase sampling mode.")

    print(f"Applying sampling for phase {phase}: mode={sampling_mode}, desired_target={desired_target}")
    X_per_phase_sampled, y_sampled = sample_classes_random(
        mode=sampling_mode, 
        X=X_per_phase, 
        desired_target=desired_target,
        label_name=label_name,
        phases=[1], # only sample attack class
        random_state=random_state
    )

    return X_per_phase_sampled, y_sampled
            

if __name__ == "__main__":
    pass