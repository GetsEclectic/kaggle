# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gaussian_process
import sklearn.linear_model as linear_model
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.tree as tree
from scipy.stats import spearmanr
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBRegressor


def create_submission_file(my_pipeline, features):
    test = read_large_csv('../input/santander-value-prediction-challenge/test.csv')
    tst_leak = pd.read_csv('../input/leaky-rows/test_leak.csv')
    test['log_leak'] = np.log1p(tst_leak['compiled_leak'])

    pipeline = create_pipeline()

    test = pipeline.transform(test)

    features = [f for f in features if f not in ['ID', 'leak', 'target']]

    preds = my_pipeline.predict(test[features])

    submission = pd.DataFrame({'ID': test.ID, 'target': np.expm1(preds)})

    print(submission.describe())

    submission.to_csv('submission.csv', index=False)


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


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5


def load_train():
    data = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")
    target = np.log1p(data.target)
    data.drop(['ID', 'target'], axis=1, inplace=True)

    leak = pd.read_csv('../input/leaky-rows/train_leak.csv')
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


def xgb_with_important_features():
    data, target = load_train()

    pipeline = create_pipeline()

    data = pipeline.fit_transform(data)

    reg = XGBRegressor(colsample_bylevel= 0.861134571342863, colsample_bytree= 0.8549681199200161, gamma= 8.711600566125166e-06, learning_rate= 0.04074349135889144, max_delta_step= 9, max_depth= 5, min_child_weight= 1, n_estimators= 1000, reg_alpha= 1.4612464562771088e-06, reg_lambda= 0.13441711873251744, scale_pos_weight= 5.136942416966736, subsample= 0.4294081846730142)
    folds = KFold(n_splits=5, shuffle=True, random_state=1)
    fold_idx = [(trn_, val_) for trn_, val_ in folds.split(data)]

    score = 0
    for trn_, val_ in fold_idx:
        reg.fit(
            data.iloc[trn_], target.iloc[trn_],
            eval_set=[(data.iloc[val_], target.iloc[val_])],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        score += rmse(target.iloc[val_], reg.predict(data.iloc[val_], ntree_limit=reg.best_ntree_limit)) / folds.n_splits

    print("rmse: " + str(score))

    # create_submission_file(reg)


def read_large_csv(filename):
    large_pd = pd.DataFrame()

    for dataframe_chunk in pd.read_csv(filename, iterator=True, chunksize=1000):
        large_pd = large_pd.append(dataframe_chunk)

    return large_pd


def lightgbm_with_important_features():
    data, target = load_train()

    pipeline = create_pipeline()

    data = pipeline.fit_transform(data)

    dtrain = lgb.Dataset(data=data,
                         label=target, free_raw_data=False)

    dtrain.construct()
    oof_preds = np.zeros(data.shape[0])

    folds = KFold(n_splits=5, shuffle=True, random_state=1)

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
        print(mean_squared_error(target.iloc[val_idx],
                                 oof_preds[val_idx]) ** .5)

    data['predictions'] = oof_preds

    print('OOF SCORE with LEAK : %9.6f'
          % (mean_squared_error(target, data['predictions']) ** .5))

    # create_submission_file(clf)


def bayes_search():
    data, target = load_train()

    pipeline = create_pipeline()

    data = pipeline.fit_transform(data)

    nrmse_scorer = make_scorer(lambda x, y: rmse(x, y) * -1)

    bayes_cv_tuner = BayesSearchCV(
        estimator = XGBRegressor(),
        search_spaces = {
            'learning_rate': (0.01, 1.0, 'log-uniform'),
            'min_child_weight': (0, 10),
            'max_depth': (0, 50),
            'max_delta_step': (0, 20),
            'subsample': (0.01, 1.0, 'uniform'),
            'colsample_bytree': (0.01, 1.0, 'uniform'),
            'colsample_bylevel': (0.01, 1.0, 'uniform'),
            'reg_lambda': (1e-9, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'gamma': (1e-9, 0.5, 'log-uniform'),
            'min_child_weight': (0, 5),
            'n_estimators': (50, 2000),
            'scale_pos_weight': (1e-6, 500, 'log-uniform')
        },
        scoring = nrmse_scorer,
        cv = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ),
        n_jobs = 1,
        n_iter = 10000,
        verbose = 0,
        refit = True,
        random_state = 42
    )


    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

        # Get current parameters and the best parameters
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest MSE: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))

        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(clf_name+"_cv_results.csv")


    # Fit the model
    result = bayes_cv_tuner.fit(data, target, callback=status_print)


class StatisticsAdder(BaseEstimator, TransformerMixin):
    columns_to_exclude = []

    def __init__(self, columns_to_exclude):
        self.columns_to_exclude = columns_to_exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = [f for f in X if f not in self.columns_to_exclude]

        X['log_of_mean'] = np.log1p(X[features].replace(0, np.nan).mean(axis=1))
        X['mean_of_log'] = np.log1p(X[features]).replace(0, np.nan).mean(axis=1)
        X['log_of_median'] = np.log1p(X[features].replace(0, np.nan).median(axis=1))
        X['nb_nans'] = X[features].isnull().sum(axis=1)
        X['the_sum'] = np.log1p(X[features].sum(axis=1))
        X['the_std'] = X[features].std(axis=1)
        X['the_kur'] = X[features].kurtosis(axis=1)
        return X


class ZeroToNaNer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.replace(0, np.nan, inplace=True)
        return X


class ImportantFeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, rmseLimit=0.7955):
        self.rmseLimit = rmseLimit
        self.feature_importance = pd.read_csv("../input/xgb-with-individual-features/feature_report.csv")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        good_features = self.feature_importance.loc[self.feature_importance['rmse'] <= self.rmseLimit].feature
        good_features = good_features.tolist() + ['log_leak']
        return X[good_features]


def create_pipeline():
    return Pipeline([
        ('filter_important_features', ImportantFeatureFilter()),
        ('add_stats', StatisticsAdder(columns_to_exclude=['ID', 'leak', 'log_leak', 'target'])),
        ('zeros_to_nans', ZeroToNaNer())
    ])


def model_comparison():
    data, target = load_train()

    pipeline = create_pipeline()

    data = pipeline.fit_transform(data)

    MLA = [
        #Ensemble Methods
        ensemble.AdaBoostRegressor(),
        ensemble.BaggingRegressor(),
        ensemble.ExtraTreesRegressor(),
        ensemble.GradientBoostingRegressor(),
        ensemble.RandomForestRegressor(),

        #Gaussian Processes
        gaussian_process.GaussianProcessRegressor(),

        #GLM
        linear_model.PassiveAggressiveRegressor(),
        linear_model.Ridge(),
        linear_model.SGDRegressor(),

        #Nearest Neighbor
        neighbors.KNeighborsRegressor(),

        #SVM
        svm.SVR(),
        svm.NuSVR(),
        svm.LinearSVR(),

        #Trees
        tree.DecisionTreeRegressor(),
        tree.ExtraTreeRegressor(),

        #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        XGBRegressor(),
        lgb.LGBMRegressor()
    ]



    #split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    #note: this is an alternative to train_test_split
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        rmse_scorer = make_scorer(rmse)
        cv_results = model_selection.cross_validate(alg, data, target, cv  = cv_split, scoring = rmse_scorer)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
        #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

        row_index+=1


    #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], inplace = True)
    MLA_compare.to_csv('mla_comparison.csv', index=True)
    print(MLA_compare)


xgb_with_important_features()