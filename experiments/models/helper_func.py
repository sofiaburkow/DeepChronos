import numpy as np
from scipy.sparse import load_npz
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn import tree


def print_feature_importances(clf, feature_names, top_k=5):
    '''
    Print the top_k feature importances from the classifier.
    Parameters:
        clf: Trained classifier with feature_importances_ attribute.
        feature_names (list): List of feature names corresponding to the features used in training.
        top_k (int): Number of top features to print.
    Returns:
        None
    '''
    importance = clf.feature_importances_
    idx = importance.argsort()[::-1][:top_k]

    print(f"Top {top_k} feature indices: {idx}")
    print(f"Number of features: {len(feature_names)}")

    for i in idx:
        print(f"{feature_names[i]}: {importance[i]:.4f}")
        

def load_datasets(dataset_dir):
    '''
    Load training and testing data from the specified dataset directory.
    Parameters:
        dataset_dir (str): Path to the dataset directory containing the data files.
    Returns:
        X_train (scipy.sparse matrix): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        X_test (scipy.sparse matrix): Testing feature matrix.
        y_test (numpy.ndarray): Testing labels.
    '''
    X_train_file_path = f"{dataset_dir}/X_train.npz"
    X_test_file_path = f"{dataset_dir}/X_test.npz"
    y_train_file_path = f"{dataset_dir}/y_train.npy"
    y_test_file_path = f"{dataset_dir}/y_test.npy"

    X_train = load_npz(X_train_file_path)
    X_test = load_npz(X_test_file_path)
    y_train = np.load(y_train_file_path, allow_pickle=True)
    y_test = np.load(y_test_file_path, allow_pickle=True)

    return X_train, y_train, X_test, y_test


def train_and_test_classifier(clf, X_train, y_train, X_test, y_test):
    '''
    Train and evaluate a classifier.
    Parameters:
        clf: Classifier instance (must implement fit and predict methods).
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Testing feature matrix.
        y_test: Testing labels.
    Returns:
        accuracy (float): Accuracy of the classifier.
        precision (float): Precision of the classifier.
        recall (float): Recall of the classifier.
        f1 (float): F1-score of the classifier.
        cm (numpy.ndarray): Confusion matrix.
    '''
    # ---- Training and Testing ----
    print("Training Classifier...")
    clf.fit(X_train, y_train)
    print("Training completed.\n")

    print("Testing Classifier...")
    y_pred = clf.predict(X_test)
    print("Testing completed.\n")

    # ---- Evaluation ----
    print("Evaluating Classifier...")
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    print("Evaluation completed.")

    return accuracy, precision, recall, f1, cm


if __name__ == "__main__":
    # Example usage
    # Command: uv run python experiments/models/helper_func.py

    # Load datasets (example with 'inside' mode)
    mode = "inside"
    scenario = "one"
    dataset_dir = f"experiments/processed_data/{mode}_split/scenario_{scenario}"
    X_train, y_train, X_test, y_test = load_datasets(dataset_dir)
    print("Data loaded:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Train and test a classifier (example with DecisionTreeClassifier)
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

    # Feature importances
    pipeline = load(f"{dataset_dir}/feature_pipeline.joblib")
    feature_names = list(pipeline.named_steps["transform"].get_feature_names_out())
    print_feature_importances(clf, feature_names, top_k=len(feature_names))