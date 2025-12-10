import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


SEED = 123


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


def sample_classes_random(mode, X, y, phases_to_sample, desired_target):
    '''
    This function performs random upsampling or downsampling on specified classes
    to reach a desired target number of samples per class.
    '''
    counts = Counter(y)
    print('Before sampling:', Counter(counts))

    sampling_strategy = {p: desired_target for p in phases_to_sample}
    print('Sampling strategy (per-phase target):', sampling_strategy)

    if mode == 'upsample':
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=SEED)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    elif mode == 'downsample': 
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=SEED)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    else:
        raise ValueError("Mode should be either 'upsample' or 'downsample'.")

    print('After sampling:', Counter(y_resampled))

    return X_resampled, y_resampled


if __name__ == "__main__":
    pass