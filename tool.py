from sklearn.model_selection import StratifiedKFold


def split(data, fold):
    features, labels = data, data.pop(data.columns[-1])
    kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    # return a generator for the train/test splits per fold
    def fold_generator():
        for train_indices, test_indices in kfold.split(features, labels):
            yield (features.iloc[train_indices], labels.iloc[train_indices]), (features.iloc[test_indices], labels.iloc[test_indices])
    return fold_generator
