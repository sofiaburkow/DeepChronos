import sys
from sklearn import tree

from helper_fun.train_func import load_datasets, train_and_test_classifier
from helper_fun.eval_func import plot_misclassified_samples
from helper_fun.xai_func import print_feature_importances

def train_and_test_decision_tree(dataset_dir, sample_weights: bool):
    '''
    Train and evaluate a Decision Tree classifier.
    '''
    # Load datasets
    X_train, y_train, y_phase_train, X_test, y_test, y_phase_test = load_datasets(dataset_dir)
    print("Data shapes:")
    print(X_train.shape, y_train.shape, y_phase_train.shape)
    print(X_test.shape, y_test.shape, y_phase_test.shape)
    print()

    # Train and test a decision tree classifier
    clf = tree.DecisionTreeClassifier()
    accuracy, precision, recall, f1, cm, y_pred = train_and_test_classifier(
        clf, X_train, y_train, X_test, y_test, sample_weights=sample_weights
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

    # Analyze results
    print_feature_importances(clf, dataset_dir)
    
    parts = dataset_dir.split('/')
    if sample_weights:
        output_dir = f"decision_tree/sample_weights/{parts[-4]}/{parts[-3]}/{parts[-2]}"
    else:
        output_dir = f"decision_tree/no_sample_weights/{parts[-4]}/{parts[-3]}/{parts[-2]}"
    plot_misclassified_samples(y_test, y_pred, y_phase_test, output_dir)


if __name__ == "__main__":
    # Command: uv run python experiments/models/decision_tree.py <dataset_dir> <true|false>
    
    if len(sys.argv) < 3:
        print("Usage: python experiments/models/decision_tree.py <dataset_dir> <sample_weights>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    sample_weights = sys.argv[2].lower() == 'true'

    train_and_test_decision_tree(dataset_dir, sample_weights)