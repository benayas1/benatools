import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import time
import category_encoders as ce
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, anneal
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import json
from abc import ABC, abstractmethod


def _get_best_params(best, params, int_params, trials):
    best_params = best.copy()
    for par_name in int_params:
        if par_name in best_params:
            best_params[par_name] = int(best_params[par_name])
    for key in params:
        if key not in best_params.keys():
            best_params[key] = params[key]
    best_n_iterations = trials.best_trial['result']['best_n_iters']
    return best_params, best_n_iterations


def _evaluation_plot(losses, best='min', invert=False, figsize=(15,6)):

    if invert:
        losses = [ x * -1 for x in losses ]
    x = np.arange(0, len(losses))
    ys = lowess(losses, x)[:, 1]
    if best == 'max':
        best_iteration = np.argmax(losses)
    else:
        best_iteration = np.argmin(losses)
    best_value = losses[best_iteration]

    plt.figure(figsize=figsize)
    plt.plot(x, ys, 'red', linewidth=3)
    plt.plot(x, losses, 'o-')
    plt.plot([best_iteration], [best_value], marker='o', markersize=10, color="red")
    plt.suptitle('Evaluation summary', fontweight='bold', fontsize=20)
    plt.title('There are %d evaluations, the best iteration is %d and the loss is %0.4f' % (len(losses), best_iteration, best_value))
    plt.show()


def _update_params(hyperopt_params, params):
    # Replace values in dict
    if params is not None:
        for k,v in params.items():
            if k in hyperopt_params:
                if v is None:
                    del hyperopt_params[k]
                else:
                    hyperopt_params[k] = v
    return hyperopt_params


def _save_json(data, path, verbose=1):
    if path[path.rfind('.'):] != '.json':
        path = path + '.json'
    with open(path, 'w') as fp:
        json.dump(data, fp)
    if verbose > 0:
        print('Best parameters saved to '+path)


def opt_catboost(X_train,
                 y_train,
                 cat_features=None,
                 cv_folds=3,
                 folds=None,
                 n_trials=20,
                 verbose=0,
                 params=None,
                 device='GPU',
                 max_rounds=5000,
                 seed=0,
                 early_stopping=20,
                 figsize=(15, 6),
                 savepath=None):
    """
    Performs n_trials using Hyperopt to obtain the best catboost parameters

    Inputs:
        X_train, y_train, cat_features: Data to build Pool object
        cv_folds: number of CV folds for validation on each trial
        folds: Custom splitting indices. This parameter has the highest priority among other data split parameters.
        n_trials: number of trials to perform
        verbose: 0 = no log and no plot, 1 = log but no plot, 2 = log and plot
        params: dict to override the Hyperopt params
        device: 'GPU' of 'CPU'
        max_rounds: max number of iterations to train every trial
        seed: random state seed
        early_stopping: early stopping rounds

    Outputs:
        dict that contains the best params and the best number of iterations
    """
    HYPEROPT_PARAMS = {'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                       'random_state': hp.choice('random_state', [0, 1, 2, 3]),
                       'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(20)),  # eg_lambda
                       'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                       'random_strength': hp.uniform('random_strength', 0.5, 3),
                       'depth': hp.quniform('depth', 2, 8, 2),  # 10,
                       'rsm': hp.uniform('rsm', 0.1, 0.9) if device != 'GPU' else None,  # colsample_bylevel
                       'loss_function': 'RMSE',
                       'eval_metric': 'RMSE',
                       'max_bin': hp.qloguniform('max_bin', np.log(32), np.log(64), 5),  # border count
                       'task_type': device # GPU devices
                        # 'devices':0 # GPU device ID
                       }

    # Replace values in dict
    HYPEROPT_PARAMS = _update_params(HYPEROPT_PARAMS, params)
    INTEGER_PARAMS_LIST = ['depth', 'max_bin', 'random_state']

    # Catboost dataset
    train_pool = cb.Pool(data=X_train, label=y_train, cat_features=cat_features)

    # If folds already provided, set cv_folds to None
    if folds:
        cv_folds = None

    # Objective function for Hyperopt
    def objective(params):
        start = time.time()

        # Control Integer Params don't go float
        for par_name in INTEGER_PARAMS_LIST:
            params[par_name] = int(np.round(params[par_name]))

        cv = cb.cv(pool=train_pool,
                   params=params,
                   iterations=max_rounds,
                   nfold=cv_folds,
                   inverted=False,
                   shuffle=True,
                   verbose=False,
                   seed=seed,
                   early_stopping_rounds=early_stopping,
                   folds=folds,
                   as_pandas=True)

        # Metric to extract the loss from
        test_loss = 'test-RMSE-mean'
        train_loss = 'train-RMSE-mean'
        best_iteration = cv[test_loss].idxmin()
        test_loss_value = cv[test_loss].iloc[best_iteration]
        train_loss_value = cv[train_loss].iloc[best_iteration]

        if verbose > 0:
            print('Train Loss: %0.4f, Test Loss: %0.4f RMSE with %d iterations. Time elapsed %s' % (
                train_loss_value, test_loss_value, best_iteration, str(round(time.time() - start,2))))

        val = {'loss': test_loss_value,  # mandatory
               'status': STATUS_OK,  # mandatory
               'best_n_iters': best_iteration}

        return val

    trials = Trials()
    hyperopt_f = fmin(fn=objective,
                      space=HYPEROPT_PARAMS,
                      algo=tpe.suggest,
                      verbose=True if verbose > 0 else False,
                      max_evals=n_trials,
                      trials=trials)

    best_params, best_n = _get_best_params(hyperopt_f, HYPEROPT_PARAMS, INTEGER_PARAMS_LIST, trials)

    # Plot results
    if verbose == 2:
        _evaluation_plot(trials.losses(), figsize=figsize)

    best = {'params': best_params, 'n': best_n}

    if savepath:
        _save_json(best , savepath)

    return best


def opt_xgboost(X_train,
                y_train,
                cat_features=None,
                cv_folds=3,
                folds=None,
                n_trials=20,
                verbose=0,
                params=None,
                device='GPU',
                max_rounds=5000,
                seed=0,
                early_stopping=20,
                figsize=(15, 6),
                savepath=None):
    """
    Performs n_trials using Hyperopt to obtain the best XGBoost parameters

    Inputs:
        X_train, y_train, cat_features: Data to build Pool object
        cv_folds: number of CV folds for validation on each trial
        folds: Custom splitting indices. This parameter has the highest priority among other data split parameters.
        n_trials: number of trials to perform
        verbose: 0 = no log and no plot, 1 = log but no plot, 2 = log and plot
        params: dict to override the Hyperopt params
        device: 'GPU' of 'CPU'
        max_rounds: max number of iterations to train every trial
        seed: random state seed
        early_stopping: early stopping rounds

    Outputs:
        dict that contains the best params and the best number of iterations
    """

    HYPEROPT_PARAMS = {'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                               'gamma': hp.quniform('gamma', 0, 0.5, 0.1),
                               'max_depth': hp.quniform('max_depth', 2, 10, 2),  # 10,
                               'min_child_weight': hp.quniform('min_child_weight', 1, 8, 2),
                               'loss_function': 'rmse',
                               'eval_metric': 'rmse',
                               'lambda': hp.loguniform('lambda', 0, np.log(20)),
                               'alpha': hp.loguniform('alpha', 0, np.log(100)),
                               'subsample': hp.uniform('subsample', 0.2, 0.8),
                               'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1, ),
                               'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                               'seed': hp.choice('seed', [0, 1, 2, 3]),
                               'gpu_id': 0,
                               'tree_method': 'gpu_hist' if device == 'GPU' else 'auto'
                               }
    # Replace values in dict
    HYPEROPT_PARAMS = _update_params(HYPEROPT_PARAMS, params)
    INTEGER_PARAMS_LIST = ['max_depth', 'seed']

    # Transform categorical features to one_hot
    if cat_features:
        X_train = ce.one_hot.OneHotEncoder(cols=cat_features, drop_invariant=True).fit_transform(X_train)
    train_dmatrix = xgb.DMatrix(X_train, y_train)

    def objective(params):
        start = time.time()
        # Control Integer Params don't go float
        for par_name in INTEGER_PARAMS_LIST:
            params[par_name] = int(np.round(params[par_name]))

        cv = xgb.cv(params=params,
                    dtrain=train_dmatrix,
                    nfold=cv_folds,
                    folds=folds,
                    stratified=False,
                    metrics=params['eval_metric'],
                    verbose_eval=False,
                    seed=seed,
                    num_boost_round=max_rounds,
                    as_pandas=True,
                    early_stopping_rounds=early_stopping)

        # Metric to extract the loss from
        test_loss = 'test-rmse-mean'
        train_loss = 'train-rmse-mean'
        best_iteration = cv[test_loss].idxmin()
        test_loss_value = cv[test_loss].iloc[best_iteration]
        train_loss_value = cv[train_loss].iloc[best_iteration]

        if verbose > 0:
            print('Train Loss: %0.4f, Test Loss: %0.4f RMSE with %d iterations. Time elapsed %s' % (
                train_loss_value, test_loss_value, best_iteration, str(round(time.time() - start,2))))

        val = {'loss': test_loss_value,  # mandatory
               'status': STATUS_OK,  # mandatory
               'best_n_iters': best_iteration}

        return val

    trials = Trials()
    hyperopt_f = fmin(fn=objective,
                      space=HYPEROPT_PARAMS,
                      algo=tpe.suggest,
                      verbose=True if verbose > 0 else False,
                      max_evals=n_trials,
                      trials=trials)

    best_params, best_n = _get_best_params(hyperopt_f, HYPEROPT_PARAMS, INTEGER_PARAMS_LIST, trials)

    # Plot results
    if verbose == 2:
        _evaluation_plot(trials.losses(), figsize=figsize)

    best = {'params': best_params, 'n': best_n}

    if savepath:
        _save_json(best, savepath)

    return best


def opt_lightgbm(X_train,
                 y_train,
                 cat_features=None,
                 cv_folds=3,
                 folds=None,
                 n_trials=20,
                 verbose=0,
                 params=None,
                 device='CPU',
                 max_rounds=5000,
                 seed=0,
                 early_stopping=20,
                 figsize=(15, 6),
                 savepath=None):
    """
    Performs n_trials using Hyperopt to obtain the best LightGBM parameters

    Inputs:
        X_train, y_train, cat_features: Data to build Pool object
        cv_folds: number of CV folds for validation on each trial
        folds: Custom splitting indices. This parameter has the highest priority among other data split parameters.
        n_trials: number of trials to perform
        verbose: 0 = no log and no plot, 1 = log but no plot, 2 = log and plot
        params: dict to override the Hyperopt params
        device: 'gpu' of 'cpu' (lowercase)
        max_rounds: max number of iterations to train every trial
        seed: random state seed
        early_stopping: early stopping rounds

    Outputs:
        dict that contains the best params and the best number of iterations
    """
    HYPEROPT_PARAMS = {'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                       'num_leaves': hp.qloguniform('num_leaves', np.log(32), np.log(256), 5),
                       'max_depth': hp.quniform('max_depth', 2, 10, 2),  # 10,
                       'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', np.log(32), np.log(256), 5),
                       # 'min_child_weight': hp.loguniform('min_child_weight', -16, 5)
                       'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 0.9),
                       'bagging_freq': hp.uniform('bagging_freq', 1, 10),
                       'feature_fraction': hp.uniform('feature_fraction', 0.2, 0.9),
                       'lambda_l1': hp.loguniform('lambda_l1', 0, np.log(100)),
                       'lambda_l2': hp.loguniform('lambda_l2', 0, np.log(20)),
                       # 'max_bin': hp.qloguniform('max_bin', np.log(32), np.log(64), 5), # border count,
                       'metric': 'rmse',
                       'seed': hp.choice('seed', [0, 1, 2, 3]),
                       'num_threads': 5,
                       # 'gpu_id': 0,
                       'device_type':device.lower()
                       }

    HYPEROPT_PARAMS = _update_params(HYPEROPT_PARAMS, params)
    INTEGER_PARAMS_LIST = ['max_depth', 'seed', 'bagging_freq','num_leaves','min_data_in_leaf']

    def objective(params):
        start = time.time()

        dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

        # Control Integer Params don't go float
        for par_name in INTEGER_PARAMS_LIST:
            params[par_name] = int(np.round(params[par_name]))

        cv = lgb.cv(params=params,
                    train_set=dataset,
                    nfold=cv_folds,
                    folds=folds,
                    stratified=False,
                    verbose_eval=False,
                    seed=seed,
                    num_boost_round=max_rounds,
                    early_stopping_rounds=early_stopping)

        # Metric to extract the loss from
        cv_results = pd.DataFrame(cv)
        metric_mean_name = 'rmse-mean'
        best_iteration = cv_results[metric_mean_name].idxmin()
        loss = cv_results[metric_mean_name].iloc[best_iteration]

        if verbose > 0:
            print('Test Loss: %0.4f RMSE with %d iterations. Time elapsed %s' % (
                loss, best_iteration, str(round(time.time() - start, 2))))

        val = {'loss': loss,  # mandatory
               'status': STATUS_OK,  # mandatory
               'best_n_iters': best_iteration}

        return val

    trials = Trials()
    hyperopt_f = fmin(fn=objective,
                      space=HYPEROPT_PARAMS,
                      algo=tpe.suggest,
                      verbose=True if verbose > 0 else False,
                      max_evals=n_trials,
                      trials=trials)

    best_params, best_n = _get_best_params(hyperopt_f, HYPEROPT_PARAMS, INTEGER_PARAMS_LIST, trials)
    # apparently hyperopt has a bug on choice methods
    # best_params['num_leaves'] = 2 ** (best_params['num_leaves']+3) - 1
    # best_params['min_data_in_leaf'] = 2 ** (best_params['min_data_in_leaf']+3) - 1
    # best_params['max_bin'] = 2 ** (best_params['max_bin']+3) - 1

    # Plot results
    if verbose == 2:
        _evaluation_plot(trials.losses(), figsize=figsize)

    best = {'params': best_params, 'n': best_n}

    if savepath:
        _save_json(best, savepath)

    return best


def _get_params(library, device='GPU'):
    if library == 'CB':
        return {'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                'random_state': hp.choice('random_state', [0, 1, 2, 3]),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(20)),  # eg_lambda
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'random_strength': hp.uniform('random_strength', 0.5, 3),
                'depth': hp.quniform('depth', 2, 8, 2),  # 10,
                'rsm': hp.uniform('rsm', 0.1, 0.9) if device != 'GPU' else None,  # colsample_bylevel
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'max_bin': hp.qloguniform('max_bin', np.log(32), np.log(64), 5),  # border count
                'task_type': device # GPU devices
                # 'devices':0 # GPU device ID
                }, ['depth', 'max_bin', 'random_state']

    if library == 'XGB':
        return {'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                'gamma': hp.quniform('gamma', 0, 0.5, 0.1),
                'max_depth': hp.quniform('max_depth', 2, 10, 2),  # 10,
                'min_child_weight': hp.quniform('min_child_weight', 1, 8, 2),
                'loss_function': 'rmse',
                'eval_metric': 'rmse',
                'lambda': hp.loguniform('lambda', 0, np.log(20)),
                'alpha': hp.loguniform('alpha', 0, np.log(100)),
                'subsample': hp.uniform('subsample', 0.2, 0.8),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1, ),
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'seed': hp.choice('seed', [0, 1, 2, 3]),
                'gpu_id': 0,
                'tree_method': 'gpu_hist' if device == 'GPU' else 'auto'
                }, ['max_depth', 'seed']

    if library == 'LGB':
        return {'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                'num_leaves': hp.qloguniform('num_leaves', np.log(32), np.log(256), 5),
                'max_depth': hp.quniform('max_depth', 2, 10, 2),  # 10,
                'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', np.log(32), np.log(256), 5),
                # 'min_child_weight': hp.loguniform('min_child_weight', -16, 5)
                'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 0.9),
                'bagging_freq': hp.uniform('bagging_freq', 1, 10),
                'feature_fraction': hp.uniform('feature_fraction', 0.2, 0.9),
                'lambda_l1': hp.loguniform('lambda_l1', 0, np.log(100)),
                'lambda_l2': hp.loguniform('lambda_l2', 0, np.log(20)),
                # 'max_bin': hp.qloguniform('max_bin', np.log(32), np.log(64), 5), # border count,
                'metric': 'rmse',
                'seed': hp.choice('seed', [0, 1, 2, 3]),
                'num_threads': 5,
                # 'gpu_id': 0,
                'device_type': device.lower()
                }, ['max_depth', 'seed', 'bagging_freq','num_leaves','min_data_in_leaf']


def get_optimizer(library, device='GPU', override_params=None):
    if library == 'CB':
        return OptimizerCB(device, override_params=override_params)
    if library == 'XGB':
        return OptimizerXGB(device, override_params=override_params)
    if library == 'LGB':
        return OptimizerLGB(device, override_params=override_params)


class BaseOptimizer(ABC):
    def __init__(self, library, device='GPU', override_params=None):
        self.library = library

        self.training_params, self.int_params = _get_params(self.library, device)
        self.training_params = _update_params(self.training_params, override_params)

    def optimize(self,
                 X_train,
                 y_train,
                 cat_features=None,
                 objective=None,
                 verbose=1,
                 max_evals=100,
                 max_rounds=5000,
                 early_stopping=50,
                 n_folds=5,
                 folds=None,
                 savepath=None):
        """
        Performs max_evals using Hyperopt to obtain the best parameters

        Inputs:
            X_train, y_train, cat_features: Data to build Pool object
            objective: objective function to minimize. If None, takes standard function
            cv_folds: number of CV folds for validation on each trial
            folds: Custom splitting indices. This parameter has the highest priority among other data split parameters.

            verbose: 0 = no log, 1 = log
            max_evals: number of trials to perform
            max_rounds: max number of iterations to train every trial
            seed: random state seed
            early_stopping: early stopping rounds
            savepath: json path to save the best results

        Outputs:
            dict that contains the best params and the best number of iterations
        """


        # Build objective function
        if objective is None:
            objective = self.get_obj(X_train, y_train, cat_features=cat_features, n_folds=n_folds, folds=folds,  max_rounds=max_rounds, early_stopping=early_stopping)

        # Run the optimization
        self.trials = Trials()
        hyperopt_f = fmin(fn=objective,
                          space=self.training_params,
                          algo=tpe.suggest,
                          verbose=True if verbose > 0 else False,
                          max_evals=max_evals,
                          trials=self.trials)

        # Extract the best params
        best_params, best_n = self._get_best_params(hyperopt_f, self.trials)

        # Builds a dict containing the best params and the best n iterations
        best = {'params': best_params, 'n': best_n}

        # Save results to json file
        if savepath:
            self._save_json(best, savepath)

        return best

    def _get_best_params(self, best, trials):
        """
        """
        best_params = best.copy()
        for par_name in self.int_params:
            if par_name in best_params:
                best_params[par_name] = int(best_params[par_name])
        for key in self.training_params:
            if key not in best_params.keys():
                best_params[key] = self.training_params[key]
        best_n_iterations = trials.best_trial['result']['best_n_iters'] if 'best_n_iters' in trials.best_trial['result'] else None
        return best_params, best_n_iterations

    def plot(self, best='min', invert=False, figsize=(15, 6)):
        if not self.trials:
            return
        losses = self.trials.losses()

        if invert:
            losses = [x * -1 for x in losses]
        x = np.arange(0, len(losses))
        ys = lowess(losses, x)[:, 1]

        best_iteration = np.argmax(losses) if best == 'max' else np.argmin(losses)

        best_value = losses[best_iteration]

        plt.figure(figsize=figsize)
        plt.plot(x, ys, 'red', linewidth=3)
        plt.plot(x, losses, 'o-')
        plt.plot([best_iteration], [best_value], marker='o', markersize=10, color="red")
        plt.suptitle('Evaluation summary', fontweight='bold', fontsize=20)
        plt.title('There are %d evaluations, the best iteration is %d and the loss is %0.4f' % (
                    len(losses), best_iteration, best_value))
        plt.show()

    def _save_json(self, data, path, verbose=1):
        if path[path.rfind('.'):] != '.json':
            path = path + '.json'
        with open(path, 'w') as fp:
            json.dump(data, fp)
        if verbose > 0:
            print('Best parameters saved to ' + path)

    @abstractmethod
    def get_obj(self, X_train, y_train, cat_features=None, n_folds=5, folds=None, max_rounds=5000, early_stopping=50, seed=0, verbose=1):
        pass


class OptimizerCB(BaseOptimizer):
    def __init__(self, device='GPU', override_params=None):
        super(OptimizerCB, self).__init__('CB', device, override_params)

    def get_obj(self, X_train, y_train, cat_features=None, n_folds=5, folds=None, max_rounds=5000, early_stopping=50, seed=0, verbose=1):
        # Catboost dataset
        train_pool = cb.Pool(data=X_train, label=y_train, cat_features=cat_features)

        # If folds already provided, set cv_folds to None
        if folds:
            n_folds = None

        # Objective function for Hyperopt
        def objective_f(params):
            start = time.time()

            # Control Integer Params don't go float
            for par_name in self.int_params:
                params[par_name] = int(np.round(params[par_name]))

            cv = cb.cv(pool=train_pool,
                       params=params,
                       iterations=max_rounds,
                       nfold=n_folds,
                       inverted=False,
                       shuffle=True,
                       verbose=False,
                       seed=seed,
                       early_stopping_rounds=early_stopping,
                       folds=folds,
                       as_pandas=True)

            # Metric to extract the loss from
            test_loss = 'test-RMSE-mean'
            train_loss = 'train-RMSE-mean'
            best_iteration = cv[test_loss].idxmin()
            test_loss_value = cv[test_loss].iloc[best_iteration]
            train_loss_value = cv[train_loss].iloc[best_iteration]

            if verbose > 0:
                print('Train Loss: %0.4f, Test Loss: %0.4f RMSE with %d iterations. Time elapsed %s' % (
                    train_loss_value, test_loss_value, best_iteration, str(round(time.time() - start, 2))))

            val = {'loss': test_loss_value,  # mandatory
                   'status': STATUS_OK,  # mandatory
                   'best_n_iters': best_iteration}

            return val
        return objective_f


class OptimizerXGB(BaseOptimizer):
    def __init__(self, device='GPU', override_params=None):
        super(OptimizerXGB, self).__init__('XGB', device, override_params)

    def get_obj(self, X_train, y_train, cat_features=None, n_folds=5, folds=None, max_rounds=5000, early_stopping=50,
                seed=0, verbose=1):

        # Transform categorical features to one_hot
        if cat_features:
            X_train = ce.one_hot.OneHotEncoder(cols=cat_features, drop_invariant=True).fit_transform(X_train)
        train_dmatrix = xgb.DMatrix(X_train, y_train)

        def objective_f(params):
            start = time.time()
            # Control Integer Params don't go float
            for par_name in self.int_params:
                params[par_name] = int(np.round(params[par_name]))

            cv = xgb.cv(params=params,
                        dtrain=train_dmatrix,
                        nfold=n_folds,
                        folds=folds,
                        stratified=False,
                        metrics=params['eval_metric'],
                        verbose_eval=False,
                        seed=seed,
                        num_boost_round=max_rounds,
                        as_pandas=True,
                        early_stopping_rounds=early_stopping)

            # Metric to extract the loss from
            test_loss = 'test-rmse-mean'
            train_loss = 'train-rmse-mean'
            best_iteration = cv[test_loss].idxmin()
            test_loss_value = cv[test_loss].iloc[best_iteration]
            train_loss_value = cv[train_loss].iloc[best_iteration]

            if verbose > 0:
                print('Train Loss: %0.4f, Test Loss: %0.4f RMSE with %d iterations. Time elapsed %s' % (
                    train_loss_value, test_loss_value, best_iteration, str(round(time.time() - start, 2))))

            val = {'loss': test_loss_value,  # mandatory
                   'status': STATUS_OK,  # mandatory
                   'best_n_iters': best_iteration}

            return val

        return objective_f


class OptimizerLGB(BaseOptimizer):
    def __init__(self, device='CPU', override_params=None):
        super(OptimizerLGB, self).__init__('LGB', device, override_params)

    def get_obj(self, X_train, y_train, cat_features=None, n_folds=5, folds=None, max_rounds=5000, early_stopping=50,
                seed=0, verbose=1):
        def objective_f(params):
            start = time.time()

            dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

            # Control Integer Params don't go float
            for par_name in self.int_params:
                params[par_name] = int(np.round(params[par_name]))

            cv = lgb.cv(params=params,
                        train_set=dataset,
                        nfold=n_folds,
                        folds=folds,
                        stratified=False,
                        verbose_eval=False,
                        seed=seed,
                        num_boost_round=max_rounds,
                        early_stopping_rounds=early_stopping)

            # Metric to extract the loss from
            cv_results = pd.DataFrame(cv)
            metric_mean_name = 'rmse-mean'
            best_iteration = cv_results[metric_mean_name].idxmin()
            loss = cv_results[metric_mean_name].iloc[best_iteration]

            if verbose > 0:
                print('Test Loss: %0.4f RMSE with %d iterations. Time elapsed %s' % (
                    loss, best_iteration, str(round(time.time() - start, 2))))

            val = {'loss': loss,  # mandatory
                   'status': STATUS_OK,  # mandatory
                   'best_n_iters': best_iteration}

            return val

        return objective_f



