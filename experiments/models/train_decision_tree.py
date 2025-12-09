import sys
from sklearn import tree

from helper_func import load_datasets, train_and_test_classifier, get_feature_names, print_feature_importances


def train_and_test_decision_tree(dataset_dir):

    # Load datasets
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
    print()

    # Feature importances
    feature_names = get_feature_names(dataset_dir)
    print_feature_importances(clf, feature_names, top_k=len(feature_names))


if __name__ == "__main__":
    # Command: uv run python experiments/models/train_decision_tree.py <dataset_dir>
    
    if len(sys.argv) < 2:
        print("Usage: python train_decision_tree.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]

    train_and_test_decision_tree(dataset_dir)
