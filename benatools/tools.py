import numpy as np
import pandas as pd
import gc


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

'''
Read Dataframe from file, based on file extension
file is a path with an extension
'''
def toDF(file):
    extension = file.split('.')[1]
    if extension == 'csv':
        return pd.read_csv(file)
    if extension == 'feather':
        return pd.read_feather(file)
    if extension == 'parquet':
        return pd.read_parquet(file)

'''
files is a list of paths
'''
def toDF_all(files):
    dfs = []
    for f in files:
        df = toDF(f)
        dfs.append(df)
    df_final = pd.concat(dfs)
    del dfs
    gc.collect()
    return df_final