# from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import StratifiedKFold

def split(data, fold):
    features, labels = data
    kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    # return a generator for the train/test splits per fold
    def fold_generator():
        for train_indices, test_indices in kfold.split(features, labels):
            yield (features.iloc[train_indices], labels.iloc[train_indices]), (features.iloc[test_indices], labels.iloc[test_indices])
    return fold_generator

def load_data(data):
    features, labels = data, data.pop('defect')
    return features, labels

def generate_random_sample(bound):
    space = bound.copy()
    for key in space:
        lower = space[key][0]
        upper = space[key][1]
        space[key] = np.float64(np.random.uniform(lower,upper,size=1))
    return space