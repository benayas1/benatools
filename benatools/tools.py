import numpy as np


class MultiStratifiedKFold():
    def __init__(self, folds, df, features):
        self.folds = folds
        self.features = features

        index = []
        for key, idx in df.groupby(self.features).groups.items():
            arr = idx.values.copy()
            np.random.shuffle(arr)
            k = (1 / self.folds)
            index.append(np.split(arr, [int(i * k * len(arr)) for i in range(1, self.folds)]))

        self.indices = list(map(np.concatenate, zip(*index)))

    def get_indices(self, fold):
        train_idx = np.concatenate([self.indices[i] for i in range(self.folds) if i != fold])
        val_idx = self.indices[fold]
        return train_idx, val_idx

    def split(self, df, fold):
        train_idx, val_idx = self.get_indices(fold)
        return df.loc[train_idx], df.loc[val_idx]

    def as_list(self):
        return [self.get_indices(i) for i in range(self.folds)]