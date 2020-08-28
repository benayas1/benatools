import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
import scipy as sp
import time
import gc
import sklearn.metrics as mt


# Class to hold and train many models (CB, XGB and LGB) on the same dataset
class GBMFitter():
    def __init__(self, cb_data=[], xgb_data=[], lgb_data=[], cv_strategy=None, cv_groups=None, use_rounders=False, metrics=['rmse']):
        """Generates models in a CV manner for all 3 libraries algorithms

        Inputs:
            - cb_data: a list of CatBoost params and iterations dicts. Syntax is [{'params':params, 'n':iteration}]
            - xgb_data: a list of XGBoost params and iterations dicts. Syntax is [{'params':params, 'n':iteration}]
            - lgb_data: a list of Light GBM params and iterations dicts. Syntax is [{'params':params, 'n':iteration}]
            - folds: Number of folds for CV
            - feature_selection: the threshold to select features. If -1, takes all
            - cv_strategy:
            - use_rounders: boolean to indicate to use rounders """
        self.training_data = {'CB': cb_data, 'XGB': xgb_data, 'LGB': lgb_data}
        self.models = {'CB': [], 'XGB': [], 'LGB': []}
        self.rounders = {'CB': [], 'XGB': [], 'LGB': []}
        self.fselection = {}
        self.cv = cv_strategy
        self.cv_groups = cv_groups
        self.use_rounders = use_rounders
        self.metrics = metrics

    def _get_categorical_index(self, df, cat_cols):
        cat_features_index = np.where(df.columns.isin(cat_cols))[0].tolist()
        return cat_features_index

    def fit(self, X, y, categorical=[], feature_selection=-1, skip_CB=False, skip_XGB=False, skip_LGB=False):
        """Generates models in a CV manner for all 3 libraries algorithms

        Inputs:
            - X: Dataframe with features
            - y: target variable
            - categorical: a list of categorical variables
            - folds: Number of folds for CV
            - feature_selection: the threshold to select features. If -1, takes all"""

        # Fits the data for each library algorithm
        if skip_CB == False:
            self._fit('CB', X, y, categorical, feature_selection)

        if skip_XGB == False:
            self._fit('XGB', X, y, categorical, feature_selection)

        if skip_LGB == False:
            self._fit('LGB', X, y, categorical, feature_selection)

    def _fit(self, library, X, y, categorical=[], feature_selection=-1):
        """ Fits data into the algorithms. Generates a model per fold, and stores a
        tuple of (model, rounder) into self.models for each fold.

        Inputs:
            - library: String indicating the library to fit the model for (CB, XGB or LGB)
            - X: Dataframe with features
            - y: target variable
            - categorical: a list of categorical variables
            - folds: Number of folds for CV
            - feature_selection: the threshold to select features. If -1, takes all"""

        if not self.training_data[library]:
            return

        print("Training " + library + " models")
        start = time.time()
        for model in self.training_data[library]:
            # Perform Feature Selection
            if feature_selection > 0:
                print("\tFeature Selection for " + library + " models")
                fselection = select_features(library)
                fselection.fit(self._train(library, model, X, y, categorical)[0], feature_selection)
                self.fselection[library] = fselection
            X_data = self.fselection[library].transform(X) if library in self.fselection else X

            # Training for CV
            if self.cv is not None:
                print("\tTraining with " + str(self.cv.get_n_splits()) + " folds")
                # Perform CV
                y_pred = np.zeros(X_data.shape[0])
                for f, (train_index, val_index) in enumerate(self.cv.split(X_data, y, self.cv_groups)):
                    print("\t\tTraining fold {} ".format(f))
                    y_pred[val_index] = self._train(library,
                                                    model,
                                                    train=(X_data.iloc[train_index], y.iloc[train_index]),
                                                    validation=(X_data.iloc[val_index], y.iloc[val_index]),
                                                    categorical=categorical)
                # Print CV metric
                for metric in self.metrics:
                    if metric == 'rmse':
                        value = mt.mean_squared_error(y, y_pred, squared=False)
                    else:
                        value = metric(y, y_pred)
                    print("\t\tOOF Validation Metric: {:.4f}, total time elapsed {}".format(value, str(round(time.time() - start, 2))))
            # If folds is 1, then train on all the dataset with no CV
            else:
                self._train(library, model, X_data, y, categorical)

    def _train(self, library, model_data, train, validation=None, categorical=[]):
        """ Trains a mode on training data, calculates predictions for training and for validation,
        and also creates and fits the corresponding OptRounder object.

        Inputs:
            - library: String indicating the library to calculate the model for (CB, XGB or LGB)
            - model_params: List with model params[0] and number of rounds[1]
            - train: Tuple with train data, train[0] is X and train[1] is y
            - validation: Validation data, validation[0] is X and validation[1] is y
            - categorical: List with the categorical variables
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
            
        # Train and predict train and validation sets
        if library == 'CB':
            m = cb.train(dtrain=cb.Pool(data=X_train,
                                        label=y_train,
                                        cat_features=self._get_categorical_index(X_train, categorical)),
                         params=model_data['params'],
                         logging_level='Silent',
                         num_boost_round=model_data['n'])
            y_pred_train = m.predict(X_train)
            y_pred_val = m.predict(X_val)

        if library == 'XGB':
            obj = None
            if 'obj' in model_data:
                obj = model_data['obj']
            m = xgb.train(params=model_data['params'],
                          dtrain=xgb.DMatrix(X_train, y_train),
                          num_boost_round=model_data['n'],
                          obj=obj)
            y_pred_train = m.predict(xgb.DMatrix(X_train))
            y_pred_val = m.predict(xgb.DMatrix(X_val))
            f = m.save_model('xgb_temp')
            m.__del__()
            gc.collect()
            m = xgb.Booster()
            m.load_model('xgb_temp')

        if library == 'LGB':
            obj = None
            if 'obj' in model_data:
                obj = model_data['obj']
            m = lgb.train(params=model_data['params'],
                          train_set=lgb.Dataset(X_train,
                                                label=y_train,
                                                free_raw_data=False),
                          categorical_feature=self._get_categorical_index(X_train,
                                                                          categorical) if categorical else 'auto',
                          num_boost_round=model_data['n'],
                          fobj=obj)
            y_pred_train = m.predict(X_train)
            y_pred_val = m.predict(X_val)

        # Fit the rounder and transform the outputs to discrete values
        if self.use_rounders:
            rounder.fit(y_pred_train, y_train)
            y_pred_train = rounder.predict(y_pred_train)
            y_pred_val = rounder.predict(y_pred_val)

        # Evaluate Train
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
        print("\t\t\tTrain Metric: {:.4f}, OOF Val Metric: {:.4f}, elapsed {}".format(train_metric, val_metric,
                                                                                  str(round(time.time() - start, 2))))

        # Store model and rounder if needed
        self.models[library].append(m)
        if self.use_rounders:
            self.rounders[library].append(rounder)

        return y_pred_val

    def predict(self, X, mean_function=None):
        """Predicts the regression value without calculating a class

        Inputs:
            - X: Data to predict a class for
            - mean_function: Function to run as part of .apply, to average the class result from all columns. It is included in a new column

        Output:
            - Returns a Pandas DataFrame with all the predicted class for each sample (row) and each model (column)"""
        df = pd.DataFrame()

        for i in range(0, len(self.models['CB'])):
            X_data = self.fselection['CB'].transform(X) if 'CB' in self.fselection else X
            df['cb' + str(i)] = self.models['CB'][i].predict(X_data)

        for i in range(0, len(self.models['XGB'])):
            X_data = self.fselection['XGB'].transform(X) if 'XGB' in self.fselection else X
            df['xgb' + str(i)] = self.models['XGB'][i].predict(xgb.DMatrix(X_data))

        for i in range(0, len(self.models['LGB'])):
            X_data = self.fselection['LGB'].transform(X) if 'LGB' in self.fselection else X
            df['lgb' + str(i)] = self.models['LGB'][i].predict(X_data)

        if not mean_function is None:
            df['mean'] = df.apply(mean_function, axis=1)

        return df

    def predict_class(self, X, mean_function=None):
        """Predicts a class for the given data X. The class is determined by the optimizer, which has been previously fitted.

        Inputs:
            - X: Data to predict a class for
            - mean_function: Function to run as part of .apply, to average the class result from all columns. It is included in a new column

        Output:
            - Returns a Pandas DataFrame with all the predicted class for each sample (row) and each model (column)"""
        df = pd.DataFrame()

        # Predict values for every calculated model for Catboost
        for i in range(0, len(self.models['CB'])):
            X_data = self.fselection['CB'].transform(X) if 'CB' in self.fselection else X
            df['cb_class' + str(i)] = self.rounders['CB'][i].predict(self.models['CB'][i].predict(X_data)).astype(int)

        # Predict values for every calculated model for XGBoost
        for i in range(0, len(self.models['XGB'])):
            X_data = self.fselection['XGB'].transform(X) if 'XGB' in self.fselection else X
            df['xgb_class' + str(i)] = self.rounders['XGB'][i].predict(
                self.models['XGB'][i].predict(xgb.DMatrix(X_data))).astype(int)

        # Predict values for every calculated model for LightGBM
        for i in range(0, len(self.models['LGB'])):
            X_data = self.fselection['LGB'].transform(X) if 'LGB' in self.fselection else X
            df['lgb_class' + str(i)] = self.rounders['LGB'][i].predict(self.models['LGB'][i].predict(X_data)).astype(
                int)

        # Calculate mean if requested
        if not mean_function is None:
            df['mean'] = df.apply(mean_function, axis=1)

        return df


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