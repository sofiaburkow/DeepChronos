import sys
from sklearn.svm import SVC

from helper_fun.train_func import load_datasets, train_and_test_classifier
from helper_fun.eval_func import plot_misclassified_samples

def train_and_test_svm(dataset_dir):
    '''
    Train and evaluate a Support Vector Machine (SVM) classifier.
    '''
    # Load datasets
    X_train, y_train, y_phase_train, X_test, y_test, y_phase_test = load_datasets(dataset_dir)
    print("Data shapes:")
    print(X_train.shape, y_train.shape, y_phase_train.shape)
    print(X_test.shape, y_test.shape, y_phase_test.shape)

    # Train and test a SVM classifier
    clf = SVC(
        kernel="rbf",
        probability=False,     
        C=1.0,
        gamma="scale"
    )
    accuracy, precision, recall, f1, cm, y_pred = train_and_test_classifier(
        clf, X_train, y_train, X_test, y_test
    )

    # Print results
    print("\n=== SVM Classifier Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()

    # Analyze results
    parts = dataset_dir.split('/')
    output_dir = f"svm/{parts[-3]}/{parts[-2]}/"
    plot_misclassified_samples(y_test, y_pred, y_phase_test, output_dir)


if __name__ == "__main__":
    # Command: uv run python experiments/models/svm.py <dataset_dir>
    
    if len(sys.argv) < 2:
        print("Usage: python svm.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]

    train_and_test_svm(dataset_dir)