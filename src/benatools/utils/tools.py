"""
Useful set of tools
"""

import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import fmin
from abc import ABC, abstractmethod
import sklearn as sk


class MultiStratifiedKFold():
    """Multi Stratified K-Fold cross validator. Indices split happens at creation time

    Parameters
    ----------
    n_splits : int
        Number of splits
    df : pd.DataFrame
        DataFrame to split
    features : list of str
        List of features to be considered when stratifying
    seed : int, defaults to 0
        Random seed
    """

    def __init__(self, n_splits, df, features, seed=0):
        self.n_splits = n_splits
        self.features = features

        index = []
        np.random.seed(seed)

        for g, (key, idx) in enumerate(df.groupby(self.features).groups.items()):
            arr = idx.values.copy()
            np.random.shuffle(arr)
            k = (1 / self.n_splits)
            splits = np.split(arr, [int(i * k * len(arr)) for i in range(1, self.n_splits)])

            if g > 0:
                # Calculate length of each split
                len_splits = np.array([len(i) for i in splits])

                # sort indexes by length
                len_idx = np.argsort(len_splits)

                tst = []
                # Calculate and sort by length of fold so far
                for j in range(len(index)):
                    tst.append([len(index[j][i]) for i in range(self.n_splits)])
                tst = np.array(tst)
                len_folds = np.flip(np.argsort(tst.sum(axis=0)))

                new_splits = [[]] * self.n_splits

                for j in range(self.n_splits):
                    source = len_idx[j]
                    dest = len_folds[j]
                    new_splits[dest] = splits[source]
            else:
                new_splits = splits

            index.append(new_splits)

        self.indices = list(map(np.concatenate, zip(*index)))  # list of tuples containing the

    def get_indices(self, fold):
        """Returns indices for a given fold in the range [0,folds)
        
        Parameters
        ----------
        fold : int
            fold index to retrieve indices from
        
        Returns
        -------
        train_idx : ndarray
            The training set indices for that split.
        test_idx : ndarray
            The testing set indices for that split.
        """
        train_idx = np.concatenate([self.indices[i] for i in range(self.n_splits) if i != fold])
        val_idx = self.indices[fold]
        return train_idx, val_idx
    
    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and tests set.
        
        Parameters
        ----------
            All parameters are included for compatibility with SK-Learn only
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        tests : ndarray
            The testing set indices for that split.
        """
        for i in range(self.n_splits):
            yield self.get_indices(i)

    def as_list(self):
        """Returns a list with the indices of each fold

        Returns
        -------
        List of tuples of ndarray, [(train_idx, val_idx)]
            List containing each split
        """
        return [self.get_indices(i) for i in range(self.n_splits)]

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        All parameters are included for compatibility with SK-Learn only

        Returns
        -------
        int :
            The number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class BaseOptimizeBlend(ABC):
    """
    Base class for Optimizer used in blending results of models.

    Parameters
    ----------
    maxiter : int
        Maximum number of iterations. Defaults to 1000
    maxfun : int
        Maximum number of function calls. Defaults to 1000
    """
    def __init__(self, maxiter=1000, maxfun=1000):
        self._coef = 0
        self._maxiter = maxiter
        self._maxfun = maxfun

    @abstractmethod
    def metric(self, coef, X, y):
        """
        This is an abstract method that shall return the metric to be minimized.
        If what is needed is to maximize the metric, then return -1 * metric.

        Parameters
        ----------
        coef : (ndarray with shape (n_models, ))
            Array of coefficients to be optimized.
        X : ( ndarray with shape (n_samples, n_models))
            Predicted values by each estimator.
        y : ( ndarray with shape (n_samples, ))
            True values.

        Returns
        -------
        float
            The score metric of all estimators with coefficients applied
        """
        coef = coef.reshape([X.shape[0]]  + [1 for i in range(len(X.shape)-1)])
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        score = sk.metrics.mean_squared_error(y, predictions)
        return score

    def fit(self, X, y):
        """
        Fit the results ``X`` to the true values ``y`` by calling ``fmin`` function

        Parameters
        ----------
        X : (ndarray with shape (n_samples, n_models))
            Predicted values by the estimators
        y : (ndarray with shape (n_samples, ))
            true values
        """
        partial_loss = partial(self.metric, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones( X.shape[0] ))
        self._coef = fmin(partial_loss, init_coef, disp=True, maxiter=self._maxiter, maxfun=self._maxfun)

    def predict(self, X):
        """
        Once the Optimizer is fitted, blend the predictions from n_models using the calculated coefficients

        Parameters
        ----------
        X : (ndarray with shape (n_samples, n_models))
            predicted values by the estimators.

        Returns
        -------
        ndarray with shape (n_samples, )
            Array with the X values blended applying coefficients
        """
        x_coef = X * self._coef
        predictions = np.sum(x_coef, axis=1)
        return predictions

    def get_coef(self):
        """
        Returns the number of splitting iterations in the cross-validator

        Returns
        -------
        ndarray with shape (n_models,)
            The calculated coefficients.
        """
        return self._coef


def read_df(file):
    """
    Read Dataframe from file, based on file extension

    Parameters
    ----------
    file : str
        Path with an extension of type ``.csv``, ``.feather`` or ``.parquet``
    """
    extension = file.split('.')[1]
    if extension == 'csv':
        return pd.read_csv(file)
    if extension == 'feather':
        return pd.read_feather(file)
    if extension == 'parquet':
        return pd.read_parquet(file)
