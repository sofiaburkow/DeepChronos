import pandas as pd
from scipy import sparse
from sklearn.inspection import permutation_importance
from joblib import load

def get_feature_names(dataset_dir):
    '''
    Load feature names from the feature pipeline.
    Parameters:
        dataset_dir (str): Path to the dataset directory containing the feature pipeline.
    Returns:
        feature_names (list): List of feature names.
    '''
    pipeline = load(f"{dataset_dir}/feature_pipeline.joblib")
    feature_names = list(pipeline.named_steps["transform"].get_feature_names_out())
    
    return feature_names


def print_feature_importances(clf, dataset_dir, top_k=5):
    '''
    Print the top_k feature importances from the classifier.
    Parameters:
        clf: Trained classifier with feature_importances_ attribute.
        dataset_dir (str): Path to the dataset directory containing the feature pipeline.
        top_k (int): Number of top features to print.
    Returns:
        None
    '''
    feature_names = get_feature_names(dataset_dir)
    importance = clf.feature_importances_
    idx = importance.argsort()[::-1][:top_k]

    print(f"Top {top_k} feature indices: {idx}")
    print(f"Number of features: {len(feature_names)}")

    for i in idx:
        print(f"{feature_names[i]}: {importance[i]:.4f}")


def print_permutation_importances(clf, X_test, y_test, dataset_dir):
    '''
    Print permutation importances for the given classifier and test data.
    Parameters:
        clf: Trained classifier.
        X_test: Test feature matrix.
        y_test: Test labels.
        dataset_dir (str): Path to the dataset directory containing the feature pipeline.
    Returns:
        None
    '''
    if sparse.issparse(X_test): # since permutation_importance does not support sparse matrices
        X_test = X_test.toarray()

    res = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=0, scoring='roc_auc', n_jobs=4)
    feat_names = getattr(X_test, "columns", [f"f{i}" for i in range(X_test.shape[1])])
    imp_df = pd.DataFrame({"feature": feat_names, "mean": res.importances_mean, "std": res.importances_std})
    imp_df.sort_values("mean", ascending=False)

    feature_names = get_feature_names(dataset_dir)

    for i in imp_df.sort_values("mean", ascending=False).head(20).itertuples():
        print(f"{feature_names[i.Index]}: {i.mean:.6f} ± {i.std:.6f}")


if __name__ == "__main__":
    pass