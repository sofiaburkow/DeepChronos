import sys
from sklearn.ensemble import RandomForestClassifier

from helper_fun.train_func import load_datasets, train_and_test_classifier
from helper_fun.eval_func import plot_misclassified_samples
from helper_fun.xai_func import print_feature_importances

def train_and_test_random_forest(dataset_dir):
    """
    Train and evaluate a Random Forest classifier.
    """
    # Load datasets
    X_train, y_train, y_phase_train, X_test, y_test, y_phase_test = load_datasets(dataset_dir)
    print("Data shapes:")
    print(X_train.shape, y_train.shape, y_phase_train.shape)
    print(X_test.shape, y_test.shape, y_phase_test.shape)

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
    accuracy, precision, recall, f1, cm, y_pred = train_and_test_classifier(
        clf, X_train, y_train, X_test, y_test, sample_weights=True
    )

    # Print results
    print("\n=== Random Forest Classifier Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()

    # Feature importances
    print_feature_importances(clf, dataset_dir)

    parts = dataset_dir.split('/')
    output_dir = f"random_forest/{parts[-4]}/{parts[-3]}/{parts[-2]}"
    plot_misclassified_samples(y_test, y_pred, y_phase_test, output_dir)


if __name__ == "__main__":
    # Command: uv run python experiments/models/random_forest.py <dataset_dir>
    
    if len(sys.argv) < 2:
        print("Usage: python random_forest.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]

    train_and_test_random_forest(dataset_dir)
