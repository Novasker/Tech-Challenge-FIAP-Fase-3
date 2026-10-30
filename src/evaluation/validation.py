"""Validation helpers: split and cross-validation grouped by municipality."""
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate


def split_by_municipality(df, group_column="id_municipio", test_size=0.2, seed=42):
    """Holds out whole municipalities, so no municipality is in both train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(df, groups=df[group_column]))
    return df.iloc[train_index].copy(), df.iloc[test_index].copy()


def cross_validate_grouped(pipeline, features_df, target, groups, n_splits=5):
    """Grouped CV. Returns mean and std per metric, for train and validation folds."""
    results = cross_validate(
        pipeline, features_df, target,
        groups=groups,
        cv=GroupKFold(n_splits=n_splits),
        scoring={"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"},
        return_train_score=True,
        n_jobs=1,
    )
    rows = []
    for metric in ["roc_auc", "f1", "accuracy"]:
        rows.append({
            "metric": metric,
            "train_mean": results[f"train_{metric}"].mean(),
            "valid_mean": results[f"test_{metric}"].mean(),
            "valid_std": results[f"test_{metric}"].std(),
        })
    return pd.DataFrame(rows)
