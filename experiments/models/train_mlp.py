import sys
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from scipy import sparse
from joblib import load

from helper_func import load_datasets, train_and_test_classifier, get_feature_names


def print_permutation_importances(clf, X_test, y_test, feature_names):
    '''
    Print permutation importances for the given classifier and test data.
    '''
    if sparse.issparse(X_test): # since permutation_importance does not support sparse matrices
        X_test = X_test.toarray()

    res = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=0, scoring='roc_auc', n_jobs=4)
    feat_names = getattr(X_test, "columns", [f"f{i}" for i in range(X_test.shape[1])])
    imp_df = pd.DataFrame({"feature": feat_names, "mean": res.importances_mean, "std": res.importances_std})
    imp_df.sort_values("mean", ascending=False)

    for i in imp_df.sort_values("mean", ascending=False).head(20).itertuples():
        print(f"{feature_names[i.Index]}: {i.mean:.6f} ± {i.std:.6f}")


def train_and_test_mlp(dataset_dir):
    '''
    Train and evaluate a two-layer Multi-Layer Perceptron (MLP) classifier.
    '''
    # Load datasets
    X_train, y_train, X_test, y_test = load_datasets(dataset_dir)

    print("Data shapes:")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
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

    # Permutation importances
    feature_names = get_feature_names(dataset_dir)
    print_permutation_importances(clf, X_test, y_test, feature_names)


if __name__ == "__main__":
    # Command: uv run python experiments/models/train_mlp.py <dataset_dir>
    
    if len(sys.argv) < 2:
        print("Usage: python train_decision_tree.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]

    train_and_test_mlp(dataset_dir)
