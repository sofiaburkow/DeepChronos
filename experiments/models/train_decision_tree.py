import sys
from sklearn import tree
from helper_func import load_datasets, train_and_test_classifier


def train_and_test_decision_tree(data_split_mode):

    # Load datasets
    dataset_dir = f"experiments/processed_data/{data_split_mode}/"
    X_train, y_train, X_test, y_test = load_datasets(dataset_dir)
    print("Data shapes:")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print()

    # Train and test a decision tree classifier
    clf = tree.DecisionTreeClassifier()
    accuracy, precision, recall, f1, cm = train_and_test_classifier(
        clf, X_train, y_train, X_test, y_test
    )

    # Print results
    print("\n=== Decision Tree Classifier Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    # Command: uv run python experiments/models/train_decision_tree.py <split|insidedmz|scenarios>
    
    if len(sys.argv) < 2:
        print("Usage: python train_mlp.py <data_split_mode>")
        print("data_split options: 'split', 'insidedmz', 'scenarios'")
        sys.exit(1)
    
    data_split_mode = sys.argv[1]  # options: split (60/40), insidedmz, scenarios
    if data_split_mode not in ['split', 'insidedmz', 'scenarios']:
        print("Invalid data_split_mode. Choose from 'split', 'insidedmz', 'scenarios'.")
        sys.exit(1)

    train_and_test_decision_tree(data_split_mode)

