# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import timeit

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from math import sqrt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew, spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from scipy.stats import describe
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb


def create_submission_file(my_pipeline):
    submission = pd.DataFrame()

    for test_data_chunk in pd.read_csv("../input/test.csv", iterator=True, chunksize=1000):
        test_data_chunk_to_predict = test_data_chunk.drop(['ID'], axis=1)
        submission_preds_chunk = my_pipeline.predict(test_data_chunk_to_predict)
        submission = submission.append(pd.DataFrame({'ID': test_data_chunk.ID, 'target': submission_preds_chunk}))

    print(submission.describe())

    submission.to_csv('submission.csv', index=False)


def rmsle_cv(model,X,y):
    rmsle = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_log_error", cv=5))
    return rmsle


class grid():
    def __init__(self,model):
        self.model = model

    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_log_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])


def drop_constant_columns(X):
    unique_df = X.nunique().reset_index()
    unique_df.columns = ["col_name", "unique_count"]
    constant_df = unique_df[unique_df["unique_count"]==1]
    return X.drop(constant_df.col_name, axis=1)


def get_high_spearman_corr_with_target(X, y):
    labels = []
    values = []
    for col in X.columns:
        if col not in ["ID", "target"]:
            labels.append(col)
            values.append(spearmanr(X[col].values, y.values)[0])

    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')
    high_corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]
    return high_corr_df


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200, evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


def kfold_cv(train_X, train_y, test_X):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    pred_test_full = 0
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
        pred_test_full += pred_test
    pred_test_full /= 5.
    pred_test_full = np.expm1(pred_test_full)
    return model, pred_test_full


def main():
    training_data = pd.read_csv("../input/train.csv")
    y = np.log1p(training_data.target)
    X = training_data.drop(['target', 'ID'], axis=1)

    cols_with_high_corr = get_high_spearman_corr_with_target(X, y).col_labels.tolist()

    X_with_high_corr = X[cols_with_high_corr]

    print(X_with_high_corr.shape)

    train_X, test_X, train_y, test_y = train_test_split(X_with_high_corr, y, test_size=0.25)

    model, pred_test_full = kfold_cv(train_X, train_y, test_X)

    print(pred_test_full)

    # my_pipeline = Pipeline([('scaler', RobustScaler()), ('rg', RandomForestRegressor(n_estimators=10))])
    #
    # my_pipeline.fit(train_X, train_y)
    #
    # preds = my_pipeline.predict(test_X)
    #
    # describe(preds)
    #
    # print("training RMSLE: " + str(sqrt(mean_squared_log_error(np.expm1(test_y), np.expm1(preds)))))

    # create_submission_file(my_pipeline)

# adding deskew took forever, aborted it
# RandomForestRegressor working better than XGBRegressor
# PCA made it much worse and took a long time
# try ensembling and/or stacking?
# grid search took forever

main()