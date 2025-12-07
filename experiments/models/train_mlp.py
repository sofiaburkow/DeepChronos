import sys
from sklearn.neural_network import MLPClassifier
from helper_func import load_datasets, train_and_test_classifier


def train_and_test_mlp(data_split_mode):
    '''
    Train and evaluate a two-layer Multi-Layer Perceptron (MLP) classifier.
    '''
    # Load datasets
    dataset_dir = f"experiments/processed_data/{data_split_mode}/"
    X_train, y_train, X_test, y_test = load_datasets(dataset_dir)

    print("Data shapes:")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

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

    accuracy, precision, recall, f1, cm = train_and_test_classifier(
        clf, X_train, y_train, X_test, y_test
    )

    # Print results
    print("\n=== MLP Classifier Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    # Command: uv run python experiments/models/train_mlp.py <split|insidedmz|scenarios>
    
    if len(sys.argv) < 2:
        print("Usage: python train_mlp.py <data_split_mode>")
        print("data_split options: 'split', 'insidedmz', 'scenarios'")
        sys.exit(1)
    
    data_split_mode = sys.argv[1]  # options: split (60/40), insidedmz, scenarios
    if data_split_mode not in ['split', 'insidedmz', 'scenarios']:
        print("Invalid data_split_mode. Choose from 'split', 'insidedmz', 'scenarios'.")
        sys.exit(1)

    train_and_test_mlp(data_split_mode)
