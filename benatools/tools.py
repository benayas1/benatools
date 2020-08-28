import numpy as np
import pandas as pd
import gc


class MultiStratifiedKFold():
    """ Multi Stratified K-Fold cross validator
    
        Indices split happens at creation time
    
    """
    def __init__(self, n_splits, df, features, seed=0):
        self.folds = n_splits
        self.features = features

        index = []
        np.random.seed(seed)

        for g, (key, idx) in enumerate(df.groupby(self.features).groups.items()):
            arr = idx.values.copy()
            np.random.shuffle(arr)
            k = (1 / self.folds)
            splits = np.split(arr, [int(i * k * len(arr)) for i in range(1, self.folds)])

            if g > 0:
                # Calculate length of each split
                len_splits = np.array([len(i) for i in splits])

                # sort indexes by length
                len_idx = np.argsort(len_splits)

                tst = []
                # Calculate and sort by length of fold so far
                for j in range(len(index)):
                    tst.append([len(index[j][i]) for i in range(self.folds)])
                tst = np.array(tst)
                len_folds = np.flip(np.argsort(tst.sum(axis=0)))

                new_splits = [[]] * self.folds

                for j in range(self.folds):
                    source = len_idx[j]
                    dest = len_folds[j]
                    new_splits[dest] = splits[source]
            else:
                new_splits = splits

            index.append(new_splits)

        self.indices = list(map(np.concatenate, zip(*index)))

    def get_indices(self, fold):
        """ Returns indices for a given fold in the range [0,folds]
        
        Parameters
        ----------
        fold: fold index to retrieve indices from
        
        Returns
        ------
        train_idx : ndarray
            The training set indices for that split.
        test_idx : ndarray
            The testing set indices for that split.
            
            """
        train_idx = np.concatenate([self.indices[i] for i in range(self.folds) if i != fold])
        val_idx = self.indices[fold]
        return train_idx, val_idx
    
    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        All parameters are included for compatibility with SK-Learn only
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        for i in range(self.folds):
            yield self.get_indices(i)

    def as_list(self):
        """ Returns a list of indices"""
        return [self.get_indices(i) for i in range(self.folds)]

    def get_n_splits(self):
        return self.folds


def to_df(file):
    """
    Read Dataframe from file, based on file extension
    file is a path with an extension
    """
    extension = file.split('.')[1]
    if extension == 'csv':
        return pd.read_csv(file)
    if extension == 'feather':
        return pd.read_feather(file)
    if extension == 'parquet':
        return pd.read_parquet(file)


def to_df_all(files):
    """
    files is a list of paths
    """
    dfs = []
    for f in files:
        df = to_df(f)
        dfs.append(df)
    df_final = pd.concat(dfs)
    del dfs
    gc.collect()
    return df_final
