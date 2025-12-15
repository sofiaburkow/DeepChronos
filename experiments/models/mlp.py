import sys
from sklearn.neural_network import MLPClassifier

from helper_fun.train_func import load_datasets, train_and_test_classifier
from helper_fun.eval_func import plot_misclassified_samples
from helper_fun.xai_func import print_permutation_importances

def train_and_test_mlp(dataset_dir):
    '''
    Train and evaluate a two-layer Multi-Layer Perceptron (MLP) classifier.
    '''
    # Load datasets
    X_train, y_train, y_phase_train, X_test, y_test, y_phase_test = load_datasets(dataset_dir)
    print("Data shapes:")
    print(X_train.shape, y_train.shape, y_phase_train.shape)
    print(X_test.shape, y_test.shape, y_phase_test.shape)
    print()

    # Train and test a MLP classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=50,          # increase if it underfits
        random_state=42,
        early_stopping=True,  # improves generalization, prevents overfitting
        n_iter_no_change=5
    )
    
    accuracy, precision, recall, f1, cm, y_pred = train_and_test_classifier(
        clf, X_train, y_train, X_test, y_test, sample_weights=True
    )
    
    # Print results
    print("\n=== MLP Classifier Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()
    
    # Analyze results
    print_permutation_importances(clf, X_test, y_test, dataset_dir)

    parts = dataset_dir.split('/')
    output_dir = f"mlp/{parts[-4]}/{parts[-3]}/{parts[-2]}"
    plot_misclassified_samples(y_test, y_pred, y_phase_test, output_dir)

    
if __name__ == "__main__":
    # Command: uv run python experiments/models/mlp.py <dataset_dir>
    
    if len(sys.argv) < 2:
        print("Usage: python experiments/models/mlp.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]

    train_and_test_mlp(dataset_dir)
