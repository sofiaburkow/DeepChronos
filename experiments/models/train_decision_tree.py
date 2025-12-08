import sys
from sklearn import tree

from helper_func import load_datasets, train_and_test_classifier, get_feature_names, print_feature_importances


def train_and_test_decision_tree(data_split_mode):

    # Load datasets
    dataset_dir = f"experiments/processed_data/{data_split_mode}"
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
    # Example command: uv run python train_decision_tree.py inside one
    
    if len(sys.argv) < 3:
        print("Usage: python train_decision_tree.py <split_mode> <scenario>")
        print("split_mode options: inside, insidedmz, scenario")
        print("scenario options: one, two")
        sys.exit(1)
    
    data_split_mode = sys.argv[1]  # options: trad_split, insidedmz, cross_scenarios
    if data_split_mode not in ['inside', 'insidedmz', 'scenario']:
        print("Invalid data_split_mode. Choose from 'inside', 'insidedmz', 'scenario'.")
        sys.exit(1)

    scenario = sys.argv[2]  # options: one, two
    if scenario not in ['one', 'two']:
        print("Invalid scenario. Choose 'one' or 'two'.")
        sys.exit(1)

    data_split_mode = f"{data_split_mode}_split/scenario_{scenario}"
    train_and_test_decision_tree(data_split_mode)
