import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
import scipy as sp
import time
import gc
import sklearn.metrics as mt
import category_encoders as ce
from functools import partial


class _Folds:
    def __init__(self, folds):
        self.indices = folds

    def split(self, X=None, y=None, groups=None):
        for i in range(len(self.indices)):
            yield self.indices[i]

    def get_n_splits(self):
        return len(self.indices)


# Class to hold and train many models (CB, XGB and LGB) on the same dataset
class GBMFitter:
    """Generates models in a CV manner for all 3 libraries algorithms

        Parameters
        ----------
        cb_data:
            list of CatBoost params and iterations dicts. Syntax is [{'params':params, 'n':iteration}]
        xgb_data :
            list of XGBoost params and iterations dicts. Syntax is [{'params':params, 'n':iteration}]
        lgb_data :
            list of Light GBM params and iterations dicts. Syntax is [{'params':params, 'n':iteration}]
        cv_strategy:
        use_rounders : boolean
            To indicate to use rounders """
    def __init__(self, cb_data=None, xgb_data=None, lgb_data=None, cv_strategy=None, cv_groups=None, use_rounders=False,
                 metrics=['rmse'], verbose=1, logfile=None):


        self.training_data = {'CB': cb_data, 'XGB': xgb_data, 'LGB': lgb_data}
        self.models = {'CB': [], 'XGB': [], 'LGB': []}  # for each category, generates folds models
        self.oof = {'CB': [], 'XGB': [], 'LGB': []}
        self.fselection = {}

        self.cv_groups = cv_groups
        self.use_rounders = use_rounders
        self.metrics = metrics

        if isinstance(cv_strategy, list):
            self.cv = _Folds(cv_strategy)
        else:
            self.cv = cv_strategy

        self.verbose = verbose
        self.logfile = logfile

    def _get_categorical_index(self, df, cat_cols=None):
        if cat_cols is None:
            return []
        cat_features_index = np.where(df.columns.isin(cat_cols))[0].tolist()
        return cat_features_index

    def fit(self, X, y, categorical=None, feature_selection=-1, skip_CB=False, skip_XGB=False, skip_LGB=False, num_boost_rounds=5000,
            early_stopping=None):
        """
        Generates models in a CV manner for all 3 libraries algorithms

        Parameters
        ----------
        X: pd.Dataframe
            Training data
        y: Series or numpy array
            Target variable
        categorical : list of str
            A list of categorical variables
        folds : int
            Number of folds for CV
        feature_selection : float
            The threshold to select features. If -1, takes all
        """

        # Fits the data for each library algorithm
        if not skip_CB:
            self._fit('CB', X, y, categorical, feature_selection, num_boost_rounds, early_stopping)

        if not skip_XGB:
            self._fit('XGB', X, y, categorical, feature_selection, num_boost_rounds, early_stopping)

        if not skip_LGB:
            self._fit('LGB', X, y, categorical, feature_selection, num_boost_rounds, early_stopping)

    def _fit(self, library, X, y, categorical=None, feature_selection=-1, num_boost_rounds=5000, early_stopping=None):
        """ Fits data into the algorithms. Generates a model per fold, and stores a
        tuple of (model, rounder) into self.models for each fold.

        Parameters
        ----------
        library: String indicating the library to fit the model for (CB, XGB or LGB)
        X: Dataframe with features
        y: target variable
        categorical: a list of categorical variables
        folds: Number of folds for CV
        feature_selection: the threshold to select features. If -1, takes all
        num_boost_rounds: Default boosting rounds
        early_stopping: Default early_stopping
        """

        if not self.training_data[library]:
            return

        self.log("Training " + library + " models")
        start = time.time()
        for model in self.training_data[library]:
            # Perform Feature Selection
            if feature_selection > 0:
                self.log("\tFeature Selection for " + library + " models")
                fselection = select_features(library)
                fselection.fit(self._train(library, model, X, y, categorical)[0], feature_selection)
                self.fselection[library] = fselection
            X_data = self.fselection[library].transform(X) if library in self.fselection else X

            # In the XGB case, categorical features must be converted to One Hot
            encoder=None
            if library=='XGB':
                if categorical:
                    encoder = ce.one_hot.OneHotEncoder(cols=categorical, drop_invariant=True).fit(X_data)
                    X_data = encoder.transform(X_data)

            # Training for CV
            if self.cv is not None:
                self.log("\tTraining with " + str(self.cv.get_n_splits()) + " folds")
                # Perform CV
                y_pred = np.zeros(X_data.shape[0])
                for f, (train_index, val_index) in enumerate(self.cv.split(X_data, y, self.cv_groups)):
                    self.log("\t\tTraining fold {} ".format(f+1))
                    y_pred[val_index] = self._train(library,
                                                    model,
                                                    train=(X_data.iloc[train_index], y.iloc[train_index]),
                                                    validation=(X_data.iloc[val_index], y.iloc[val_index]),
                                                    categorical=categorical,
                                                    num_boost_rounds=num_boost_rounds,
                                                    early_stopping=early_stopping,
                                                    encoder=encoder)
                # Print CV metric
                for metric in self.metrics:
                    if metric == 'rmse':
                        value = mt.mean_squared_error(y, y_pred, squared=False)
                    else:
                        value = metric(y, y_pred)
                    self.log("\t\tOOF Validation Metric: {:.4f}, total time elapsed {}".format(value, str(
                        round(time.time() - start, 2))))

                # save OOF results
                self.oof[library].append(y_pred)

            # If folds is 1, then train on all the dataset with no CV
            else:
                self._train(library, model, X_data, y, categorical)  # TODO to be tested

    def _train(self, library, model_data, train, validation=None, categorical=[], num_boost_rounds=5000, early_stopping=None, encoder=None):
        """
        Trains a mode on training data, calculates predictions for training and for validation,
        and also creates and fits the corresponding OptRounder object.

        Parameters
        ----------
        library : str
            String indicating the library to calculate the model for (CB, XGB or LGB)
        model_params :
            List with model params[0] and number of rounds[1]
        train : tuple of (pd.DataFrame, Series or numpy array)
            Tuple with train data, train[0] is X and train[1] is y
        validation : tuple of (pd.DataFrame, Series or numpy array), optional
            Validation data, validation[0] is X and validation[1] is y
        categorical : list of str
            List with the categorical variables, optional
        num_boost_rounds : int, defaults to 5000
            Default boosting rounds. Could be overriden by individual values in model_data['n]
        early_stopping : int
            Default early_stopping. Could be overriden by individual values in model_data['es']
        """

        # Get train and validation sets
        start = time.time()
        X_train = train[0]
        y_train = train[1]
        if validation is not None:
            X_val = validation[0]
            y_val = validation[1]

        if self.use_rounders:
            rounder = OptRounder()

        obj = model_data['obj'] if 'obj' in model_data else None  # Objective function
        num_boost_rounds = model_data['n'] if 'n' in model_data else num_boost_rounds  # Num rounds
        early_stopping = model_data['es'] if 'es' in model_data else early_stopping  # Early Stopping

        # Train and predict train and validation sets
        if library == 'CB':
            m = cb.train(dtrain=cb.Pool(data=X_train,
                                        label=y_train,
                                        cat_features=self._get_categorical_index(X_train, categorical)),
                         params=model_data['params'],
                         num_boost_round=num_boost_rounds,
                         early_stopping_rounds=early_stopping,
                         eval_set=cb.Pool(data=X_val,
                                          label=y_val,
                                          cat_features=self._get_categorical_index(X_train,
                                                                                   categorical)) if early_stopping else None,
                         verbose_eval=False)
            y_pred_train = m.predict(X_train)
            y_pred_val = m.predict(X_val)

        if library == 'XGB':
            m = xgb.train(params=model_data['params'],
                          dtrain=xgb.DMatrix(X_train, y_train),
                          num_boost_round=num_boost_rounds,
                          early_stopping_rounds=early_stopping,
                          evals=[(xgb.DMatrix(X_train, y_train), 'Validation')] if early_stopping else [],
                          verbose_eval=False,
                          obj=obj)
            y_pred_train = m.predict(xgb.DMatrix(X_train))
            y_pred_val = m.predict(xgb.DMatrix(X_val))
            f = m.save_model('xgb_temp')
            m.__del__()  # release memory as XGB does not do it automatically
            gc.collect()
            m = xgb.Booster()
            m.load_model('xgb_temp')

        if library == 'LGB':
            m = lgb.train(params=model_data['params'],
                          train_set=lgb.Dataset(X_train,
                                                label=y_train,
                                                free_raw_data=False),
                          categorical_feature=self._get_categorical_index(X_train,
                                                                          categorical) if categorical else 'auto',
                          num_boost_round=num_boost_rounds,
                          fobj=obj,
                          early_stopping_rounds=early_stopping,
                          valid_sets=[lgb.Dataset(X_val,
                                                  label=y_val,
                                                  free_raw_data=False)] if early_stopping else None,
                          valid_names=['Validation'] if early_stopping else None,
                          verbose_eval=False,
                          )
            # Predict validation dataset to get train result and OOF validation result
            y_pred_train = m.predict(X_train)
            y_pred_val = m.predict(X_val)

        # Fit the rounder and transform the outputs to discrete values
        if self.use_rounders:
            rounder.fit(y_pred_train, y_train)
            y_pred_train = rounder.predict(y_pred_train)
            y_pred_val = rounder.predict(y_pred_val)

        # Calculate metrics
        for metric in self.metrics:
            if metric == 'rmse':
                train_metric = mt.mean_squared_error(y_train, y_pred_train, squared=False)
                val_metric = mt.mean_squared_error(y_val, y_pred_val, squared=False)
            else:
                train_metric = metric(y_train, y_pred_train)
                val_metric = metric(y_val, y_pred_val)

        # acc_train = accuracy_score(y_train, y_pred_train)
        # f1_train = f1_score(y_train, y_pred_train, average='macro')

        # Evaluate Validation
        # acc = accuracy_score(y_val, y_pred_val)
        # f1 = f1_score(y_val, y_pred_val, average='macro')

        # print("\t\t\tTrain Accuracy: {:.4f}, Train F1: {:.4f}, Val Accuracy: {:.4f}, Val F1: {:.4f},  elapsed {}".format( acc_train, f1_train, acc, f1, str(time.time() - start)) )
        self.log("\t\t\tTrain Metric: {:.4f}, OOF Val Metric: {:.4f}, elapsed {}".format(train_metric, val_metric,
                                                                                         str(round(time.time() - start,
                                                                                                   2))))

        # Store model and rounder if needed
        data = {'m':m}
        if self.use_rounders:
            data['rounder'] = rounder
        if encoder:
            data['encoder'] = encoder
        self.models[library].append(data)


        return y_pred_val

    def get_oof(self, library):
        """
        Returns OOF for

        Parameters
        ----------
        library : str
            Whether 'CB', 'XGB' or 'LGB'

        Returns
        -------
        list of ndarray
            One array per configuration"""
        return self.oof[library]

    def predict(self, X, categorical=None, mean_function=None):
        """
        Predicts the regression value without calculating a class

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict a class for
        mean_function : function
            Function to run as part of .apply, to average the class result from all columns. It is included in a new column

        Returns
        -------
        pd.DataFrame
            All the predicted class for each sample (row) and each model (column)"""
        df = pd.DataFrame()

        for i in range(0, len(self.models['CB'])):
            X_data = self.fselection['CB'].transform(X) if 'CB' in self.fselection else X
            df['cb' + str(i)] = self.models['CB'][i]['m'].predict(X_data)

        for i in range(0, len(self.models['XGB'])):
            X_data = self.fselection['XGB'].transform(X) if 'XGB' in self.fselection else X

            if categorical:
                encoder = self.models['XGB'][i]['encoder']
                X_data = encoder.transform(X_data)

            df['xgb' + str(i)] = self.models['XGB'][i]['m'].predict(xgb.DMatrix(X_data))

        for i in range(0, len(self.models['LGB'])):
            X_data = self.fselection['LGB'].transform(X) if 'LGB' in self.fselection else X
            df['lgb' + str(i)] = self.models['LGB'][i]['m'].predict(X_data)

        if not mean_function is None:
            df['mean'] = df.apply(mean_function, axis=1)

        return df

    def predict_class(self, X, mean_function=None):
        """Predicts a class for the given data X. The class is determined by the optimizer, which has been previously fitted.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict a class for
        mean_function : function
            Function to run as part of .apply, to average the class result from all columns. It is included in a new column

        Returns
        -------
        pd.DataFrame
            All the predicted class for each sample (row) and each model (column)"""
        df = pd.DataFrame()

        # Predict values for every calculated model for Catboost
        for i in range(0, len(self.models['CB'])):
            X_data = self.fselection['CB'].transform(X) if 'CB' in self.fselection else X
            rounder = self.models['XGB'][i]['rounders']
            df['cb_class' + str(i)] = rounder.predict(self.models['CB'][i]['m'].predict(X_data)).astype(int)

        # Predict values for every calculated model for XGBoost
        for i in range(0, len(self.models['XGB'])):
            X_data = self.fselection['XGB'].transform(X) if 'XGB' in self.fselection else X
            rounder = self.models['XGB'][i]['rounders']
            df['xgb_class' + str(i)] = rounder.predict(self.models['XGB'][i]['m'].predict(xgb.DMatrix(X_data))).astype(int)

        # Predict values for every calculated model for LightGBM
        for i in range(0, len(self.models['LGB'])):
            X_data = self.fselection['LGB'].transform(X) if 'LGB' in self.fselection else X
            rounder = self.models['XGB'][i]['rounders']
            df['lgb_class' + str(i)] = rounder.predict(self.models['LGB'][i]['m'].predict(X_data)).astype(int)

        # Calculate mean if requested
        if not mean_function is None:
            df['mean'] = df.apply(mean_function, axis=1)

        return df

    def log(self, message):
        """
        Log training ouput into console and file

        Parameters
        ----------
        message : str
            message to be logged
        """
        if self.verbose > 0:
            print(message)
        if self.logfile:
            with open(self.logfile, 'a+') as logger:
                logger.write(f'{message}\n')


def select_features(library):
    if library == 'CB':
        return SelectFeaturesCB()
    if library == 'XGB':
        return SelectFeaturesXGB()
    if library == 'LGB':
        return SelectFeaturesLGB()


# Base class for Feature selection
class SelectFeatures():
    def transform(self, df):
        return df[self.features]


class SelectFeaturesCB(SelectFeatures):
    def fit(self, model, threshold):
        lista = model.get_feature_importance(prettified=True)
        lista['Importances'] = lista['Importances'] / 100
        lista['cum'] = lista['Importances'].cumsum()
        self.features = list(lista[lista['cum'] < threshold]['Feature Id'])
        print('\tNumber of selected features for CB: ' + str(len(self.features)))


class SelectFeaturesXGB(SelectFeatures):
    def fit(self, model, threshold):
        lista = list(model.get_score(importance_type='gain').items())
        suma = sum([x[1] for x in lista])
        lista = [(x[0], x[1] / suma) for x in lista]
        lista.sort(key=lambda x: x[1], reverse=True)
        v = 0
        i = 0
        while v < threshold:
            v = v + lista[i][1]
            i = i + 1
        lista = [x[0] for x in lista]
        self.features = lista[:i - 1]
        print('\tNumber of selected features for XGB: ' + str(len(self.features)))


class SelectFeaturesLGB(SelectFeatures):
    def fit(self, model, threshold):
        names = model.feature_name()
        imp = model.feature_importance(importance_type='gain')
        lista = [(names[i], imp[i] / imp.sum()) for i in range(len(names))]
        lista.sort(key=lambda x: x[1], reverse=True)
        v = 0
        i = 0
        while v < threshold:
            v = v + lista[i][1]
            i = i + 1
        lista = [x[0] for x in lista]
        self.features = lista[:i - 1]
        print('\tNumber of selected features for LGB: ' + str(len(self.features)))


class OptRounder(object):
    def __init__(self):
        self.res_ = []
        self.coef_ = []

    def get_res(self):
        return self.res_

    # objective function to the solver
    def func(self, coef, X, y):
        f1 = f1_score(self.bincut(coef, X), y, average='macro')
        return -f1

    def bincut(self, coef, X):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[i for i in range(11)])

    def fit(self, X, y):
        pfunc = partial(self.func, X=X, y=y)
        self.res_ = sp.optimize.minimize(fun=pfunc,  # objective func
                                         x0=[i + 0.5 for i in range(10)],  # initial coef
                                         method='nelder-mead')  # solver
        self.coef_ = self.res_.x

    def predict(self, y, coef=None):
        if coef is None:
            coef = self.coef_
        return self.bincut(coef, y)


def voting(x):
    votes = [0] * 11
    for i, v in x.iteritems():
        votes[v] = votes[v] + 1
    return np.asarray(votes).argmax()