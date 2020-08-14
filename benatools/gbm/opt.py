import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import time
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, anneal
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def get_best_params(hyperopt, params, int_params, trials):
    best_params = hyperopt.copy()
    for par_name in int_params:
        best_params[par_name] = int(best_params[par_name])
    for key in params:
        if key not in best_params.keys():
            best_params[key] = params[key]
    best_n_iterations = trials.best_trial['result']['best_n_iters']
    return best_params, best_n_iterations

def evaluation_plot(losses, best='min', invert=False):

    if invert == True:
        losses = [ x * -1 for x in losses ]
    x = np.arange(0, len(losses))
    ys = lowess(losses, x)[:, 1]
    if best == 'max':
        best_iteration = np.argmax(losses)
    else:
        best_iteration = np.argmin(losses)
    best_value = losses[best_iteration]

    plt.figure(figsize=(15, 6))
    plt.plot(x, ys, 'red', linewidth=3)
    plt.plot(x, losses, 'o-')
    plt.plot([best_iteration], [best_value], marker='o', markersize=10, color="red")
    plt.suptitle('Evaluation summary', fontweight='bold', fontsize=20)
    plt.title('There are %d evaluations, the best iteration is %d and the loss is %0.4f' % (len(losses), best_iteration, best_value))
    plt.show()

def opt_catboost(X_train, y_train, cat_features=None, cv_folds=3, n_trials=20, verbose=0, params=None, device='GPU', max_rounds=5000, seed=0, early_stopping=20):
    """
    Performs n_trials using Hyperopt to obtain the best catboost parameters

    Inputs:
        X_train, y_train, cat_features: Data to build Pool object
        cv_folds: number of CV folds for validation on each trial
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
                       'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(20)), #reg_lambda
                       'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                       'random_strength': hp.uniform('random_strength', 0.5, 3),
                       'depth': hp.quniform('depth', 2, 8, 2),  # 10,
                       'rsm': hp.uniform('rsm', 0.1, 0.9), #colsample_bylevel
                       'loss_function': 'RMSE',
                       'eval_metric': 'RMSE',
                       'max_bin': hp.qloguniform('border_count', np.log(32), np.log(64), 5), #border count
                       'od_type': 'Iter',  # 'IncToDec',  # 'Iter'
                       'od_wait': early_stopping,
                       'thread_count': -1,
                       'task_type': device # GPU devices
                        # 'devices':0 # GPU device ID
                       }

    # Replace values in dict
    for k,v in params.items():
        if k in HYPEROPT_PARAMS:
            HYPEROPT_PARAMS[k] = v

    INTEGER_PARAMS_LIST = ['depth', 'max_bin', 'random_state', 'thread_count']

    # Catboost dataset
    train_pool = cb.Pool(data=X_train, label=y_train, cat_features=cat_features)

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
                   as_pandas=True)

        # Metric to extract the loss from
        test_loss = 'test-RMSE-mean'
        train_loss = 'train-RMSE-mean'
        best_iteration = cv[test_loss].idxmin()
        test_loss_value = cv[test_loss].iloc[best_iteration]
        train_loss_value = cv[train_loss].iloc[best_iteration]

        if verbose > 0:
            print('Train Loss: %0.4f, Test Loss: %0.4f RMSE with %d iterations. Time elapsed %s' % (
            train_loss_value, test_loss_value, best_iteration, str(time.time() - start)))

        val = {'loss': test_loss_value,  # mandatory
               'status': STATUS_OK,  # mandatory
               'best_n_iters': best_iteration}

        return val

    trials = Trials()
    hyperopt_cb = fmin(fn=objective,
                       space=HYPEROPT_PARAMS,
                       algo=tpe.suggest,
                       verbose=True,
                       max_evals=50,
                       trials=trials)

    best_cb, best_n_cb = get_best_params(hyperopt_cb, HYPEROPT_PARAMS, INTEGER_PARAMS_LIST)

    # Plot results
    if verbose == 2:
        evaluation_plot(trials.losses())

    return {'params':best_cb, 'n':best_n_cb}

