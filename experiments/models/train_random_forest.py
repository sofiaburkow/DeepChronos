import sys
from sklearn.ensemble import RandomForestClassifier
from joblib import load

from helper_func import load_datasets, train_and_test_classifier, print_feature_importances


def train_and_test_random_forest(data_split_mode):
    """
    Train and evaluate a Random Forest classifier.
    """
    # Load datasets
    dataset_dir = f"experiments/processed_data/{data_split_mode}"
    X_train, y_train, X_test, y_test = load_datasets(dataset_dir)

    print("Data shapes:")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Train and test a Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=200,        # number of trees
        max_depth=None,          # allow full growth
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,               # use all CPU cores
        random_state=42
    )

    accuracy, precision, recall, f1, cm = train_and_test_classifier(
        clf, X_train, y_train, X_test, y_test
    )

    # Print results
    print("\n=== Random Forest Classifier Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Feature importances
    pipeline = load(f"{dataset_dir}/feature_pipeline.joblib")
    feature_names = list(pipeline.named_steps["transform"].get_feature_names_out())
    print(feature_names)
    print(len(feature_names))
    print_feature_importances(clf, feature_names, top_k=len(feature_names))


if __name__ == "__main__":
    # Command: uv run python experiments/models/train_random_forest.py <split|insidedmz|scenarios>
    
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
    train_and_test_random_forest(data_split_mode)
