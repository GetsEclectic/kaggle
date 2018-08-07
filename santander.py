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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error


def create_submission_file(my_pipeline):
    submission = pd.DataFrame()

    for test_data_chunk in pd.read_csv("../input/santander-value-prediction-challenge/test.csv", iterator=True, chunksize=1000):
        test_data_chunk_to_predict = test_data_chunk.drop(['ID'], axis=1)
        submission_preds_chunk = my_pipeline.predict(test_data_chunk_to_predict)
        submission = submission.append(pd.DataFrame({'ID': test_data_chunk.ID, 'target': submission_preds_chunk}))

    print(submission.describe())

    submission.to_csv('submission.csv', index=False)


def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


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
        dev_X, val_X = train_X.iloc[dev_index], train_X.iloc[val_index]
        dev_y, val_y = train_y.iloc[dev_index], train_y.iloc[val_index]
        pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
        pred_test_full += pred_test
    pred_test_full /= 5.
    pred_test_full = np.expm1(pred_test_full)
    return model, pred_test_full


def main():
    training_data = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")
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


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5


def load_train():
    data = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")
    target = np.log1p(data.target)
    data.drop(['ID', 'target'], axis=1, inplace=True)

    leak = pd.read_csv('../input/leaky-rows/train_leak.csv')
    data['leak'] = leak['compiled_leak'].values
    data['log_leak'] = np.log1p(leak['compiled_leak'].values)

    return data, target


def calculate_feature_importance():
    data, target = load_train()

    reg = XGBRegressor(n_estimators=1000)
    folds = KFold(4, True, 134259)
    fold_idx = [(trn_, val_) for trn_, val_ in folds.split(data)]
    scores = []

    nb_values = data.nunique(dropna=False)
    nb_zeros = (data == 0).astype(np.uint8).sum(axis=0)

    features = [f for f in data.columns if f not in ['log_leak', 'leak', 'target', 'ID']]
    for _f in features:
        score = 0
        for trn_, val_ in fold_idx:
            reg.fit(
                data[['log_leak', _f]].iloc[trn_], target.iloc[trn_],
                eval_set=[(data[['log_leak', _f]].iloc[val_], target.iloc[val_])],
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=False
            )
            score += rmse(target.iloc[val_], reg.predict(data[['log_leak', _f]].iloc[val_], ntree_limit=reg.best_ntree_limit)) / folds.n_splits
        scores.append((_f, score))

    report = pd.DataFrame(scores, columns=['feature', 'rmse']).set_index('feature')
    report['nb_zeros'] = nb_zeros
    report['nunique'] = nb_values
    report.sort_values(by='rmse', ascending=True, inplace=True)

    report.to_csv('feature_report.csv', index=True)


def read_large_csv(filename):
    large_pd = pd.DataFrame()

    for dataframe_chunk in pd.read_csv(filename, iterator=True, chunksize=1000):
        large_pd = large_pd.append(dataframe_chunk)

    return large_pd

def lightgbm_with_important_features():
    data, target = load_train()

    feature_importance = pd.read_csv("../input/xgb-with-individual-features/feature_report.csv")
    good_features = feature_importance.loc[feature_importance['rmse'] <= 0.7955].feature

    print("before test load")
    test = read_large_csv('../input/santander-value-prediction-challenge/test.csv')
    print("after test load")

    tst_leak = pd.read_csv('../input/leaky-rows/test_leak.csv')
    test['leak'] = tst_leak['compiled_leak']
    test['log_leak'] = np.log1p(tst_leak['compiled_leak'])

    folds = KFold(n_splits=5, shuffle=True, random_state=1)

    # Use all features for stats
    features = [f for f in data if f not in ['ID', 'leak', 'log_leak', 'target']]
    data.replace(0, np.nan, inplace=True)
    data['log_of_mean'] = np.log1p(data[features].replace(0, np.nan).mean(axis=1))
    data['mean_of_log'] = np.log1p(data[features]).replace(0, np.nan).mean(axis=1)
    data['log_of_median'] = np.log1p(data[features].replace(0, np.nan).median(axis=1))
    data['nb_nans'] = data[features].isnull().sum(axis=1)
    data['the_sum'] = np.log1p(data[features].sum(axis=1))
    data['the_std'] = data[features].std(axis=1)
    data['the_kur'] = data[features].kurtosis(axis=1)

    test.replace(0, np.nan, inplace=True)
    test['log_of_mean'] = np.log1p(test[features].replace(0, np.nan).mean(axis=1))
    test['mean_of_log'] = np.log1p(test[features]).replace(0, np.nan).mean(axis=1)
    test['log_of_median'] = np.log1p(test[features].replace(0, np.nan).median(axis=1))
    test['nb_nans'] = test[features].isnull().sum(axis=1)
    test['the_sum'] = np.log1p(test[features].sum(axis=1))
    test['the_std'] = test[features].std(axis=1)
    test['the_kur'] = test[features].kurtosis(axis=1)

    # Only use good features, log leak and stats for training
    features = good_features.tolist()
    features = features + ['log_leak', 'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 'the_sum', 'the_std', 'the_kur']
    dtrain = lgb.Dataset(data=data[features],
                         label=target, free_raw_data=False)
    test['target'] = 0

    dtrain.construct()
    oof_preds = np.zeros(data.shape[0])

    for trn_idx, val_idx in folds.split(data):
        lgb_params = {
            'objective': 'regression',
            'num_leaves': 58,
            'subsample': 0.6143,
            'colsample_bytree': 0.6453,
            'min_split_gain': np.power(10, -2.5988),
            'reg_alpha': np.power(10, -2.2887),
            'reg_lambda': np.power(10, 1.7570),
            'min_child_weight': np.power(10, -0.1477),
            'verbose': -1,
            'seed': 3,
            'boosting_type': 'gbdt',
            'max_depth': -1,
            'learning_rate': 0.05,
            'metric': 'l2',
        }

        clf = lgb.train(
            params=lgb_params,
            train_set=dtrain.subset(trn_idx),
            valid_sets=dtrain.subset(val_idx),
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=0
        )

        oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
        test['target'] += clf.predict(test[features]) / folds.n_splits
        print(mean_squared_error(target.iloc[val_idx],
                                 oof_preds[val_idx]) ** .5)

    data['predictions'] = oof_preds
    data.loc[data['leak'].notnull(), 'predictions'] = np.log1p(data.loc[data['leak'].notnull(), 'leak'])
    print('OOF SCORE : %9.6f'
          % (mean_squared_error(target, oof_preds) ** .5))
    print('OOF SCORE with LEAK : %9.6f'
          % (mean_squared_error(target, data['predictions']) ** .5))

    submission = pd.DataFrame({'ID': test.ID, 'target': test.target.apply(np.expm1)})

    print(submission.describe())

    submission.to_csv('submission.csv', index=False)


lightgbm_with_important_features()

# adding deskew took forever, aborted it
# RandomForestRegressor working better than XGBRegressor
# PCA made it much worse and took a long time
# try ensembling and/or stacking?
# grid search took forever

