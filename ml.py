from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import openpyxl
import time

sns.set()
matplotlib.use('Qt5Agg', force=True)

def initialize_classifiers(model_names: list):
    param_grid_Ridge = [{'solver': ['svd', 'cholesky', 'lsqr'], 'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}]
    param_grid_knn = [{}, {'n_neighbors': [1, 2, 3, 4]}]
    param_grid_LR = [{'fit_intercept': [True, False]}]
    param_grid_tree = [{'random_state': [18]},
                       {'max_depth': [2, 3], 'min_samples_split': [3, 5]}]
    param_grid_rf = [{'random_state': [18]},
                     {'n_estimators': [10, 50], 'max_features': [0.2, 0.3], 'bootstrap': [True]}]
    param_grid_gb = [{'random_state': [18]},
                     {'n_estimators': [10, 50], 'max_features': [0.2, 0.3]}]

    model_names = ['RidgeClassifier', 'KNN', 'LR', 'Tree', 'RF', 'Gradient Boost']

    return ([(RidgeClassifier(), model_names[0], param_grid_Ridge),
             (KNeighborsClassifier(), model_names[1], param_grid_knn),
             (LogisticRegression(), model_names[2], param_grid_LR),
             (DecisionTreeClassifier(), model_names[3], param_grid_tree),
             (RandomForestClassifier(), model_names[4], param_grid_rf),
             (GradientBoostingClassifier(), model_names[5], param_grid_gb)])


def initialize_regressors(model_names: list):
    param_grid_Ridge = [{'solver': ['svd', 'cholesky', 'lsqr'], 'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}]
    param_grid_Lasso = [{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'fit_intercept': [True, False]}]
    param_grid_knn = [{}, {'n_neighbors': [1, 2, 3, 4]}]
    param_grid_LR = [{'fit_intercept': [True, False]}]
    param_grid_tree = [{'random_state': [18]},
                       {'max_depth': [2, 3], 'min_samples_split': [3, 5]}]
    param_grid_rf = [{'random_state': [18]},
                     {'n_estimators': [10, 50], 'max_features': [0.2, 0.3], 'bootstrap': [True]}]
    param_grid_gb = [{'random_state': [18]},
                     {'n_estimators': [10, 50], 'max_features': [0.2, 0.3]}]

    return ([(Ridge(), model_names[0], param_grid_Ridge),
             (Lasso(), model_names[1], param_grid_Lasso),
             (KNeighborsRegressor(), model_names[2], param_grid_knn),
             (LinearRegression(), model_names[3], param_grid_LR),
             (DecisionTreeRegressor(), model_names[4], param_grid_tree),
             (RandomForestRegressor(), model_names[5], param_grid_rf),
             (GradientBoostingRegressor(), model_names[6], param_grid_gb)])


def plot_learning_curve(estimator, title, X, y, results, ylim=None, cv=None, test_score=None,
                        n_jobs=None, train_sizes=np.linspace(0.1, 1, 5), grid_search: bool = True):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].set_ylabel('Score')
    ax[0].set_xlabel('Training examples')
    if ylim is not None:
        ax[0].set_ylim(*ylim)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator=estimator, X=X, y=y, cv=cv,
                                                                          n_jobs=n_jobs, train_sizes=train_sizes,
                                                                          return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    ax[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std, alpha=0.1,
                       color='r')

    ax[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                       test_scores_mean + test_scores_std, alpha=0.1,
                       color='g')

    ax[0].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    ax[0].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross Validation Score')

    if grid_search:
        ax[0].set_title(f'{title}\n{results.best_score_}\n{results.best_estimator_}')
    else:
        if test_score is not None:
            ax[0].set_title(f'{title}\n{test_score}')
        else:
            ax[0].set_title(title)

    ax[0].legend(loc='best').set_draggable(state=True)

    ax[1].plot(train_sizes, fit_times_mean, 'o-')
    ax[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                       fit_times_mean + fit_times_std, alpha=0.1)
    ax[1].set_xlabel('Training Examples')
    ax[1].set_ylabel('Fit Times')
    ax[1].set_title('Scalability of the model')

    ax[2].plot(fit_times_mean, test_scores_mean)
    ax[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                       test_scores_mean + test_scores_std, alpha=0.1)
    ax[2].set_xlabel('Fit Times')
    ax[2].set_ylabel('Test Score')
    ax[2].set_title('Performance of the model')


def learn(df, indep_vars, dep_var, regression: bool = True, plot: bool = True):
    learning_time = time.time()
    learning_rows = df.shape[0]
    X = df[indep_vars].values
    y = df[dep_var].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=18)
    model_names = ['Ridge', 'Lasso', 'KNN', 'LR', 'Tree', 'RF', 'Gradient Boost']
    scores = []
    for i in range(len(model_names)):
        if regression:
            estimator = initialize_regressors(model_names=model_names)[i][0]
            param_grid = initialize_regressors(model_names=model_names)[i][2]
        else:
            estimator = initialize_classifiers(model_names=model_names)[i][0]
            param_grid = initialize_classifiers(model_names=model_names)[i][2]

        title = model_names[i]
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='r2')
        results = clf.fit(X, y)
        estimator = results.best_estimator_
        scores.append((results.best_score_, estimator))
        if plot:
            plot_learning_curve(estimator=estimator, title=title, X=X, y=y, cv=None, n_jobs=4, results=results)

    max_score = -999999
    for score in range(len(scores)):
        if scores[score][0] > max_score:
            max_score = scores[score][0]
            best_estimator = scores[score][1]
            model_name = model_names[score]

    clf = best_estimator
    results = clf.fit(X_train, y_train)
    print(f'Machine Learning took {time.time() - learning_time} seconds for {learning_rows} rows of data')
    test_score = results.score(X_test, y_test)
    plot_learning_curve(estimator=best_estimator, title=model_name, X=X, y=y, cv=None, n_jobs=4, results=results,
                            grid_search=False, test_score=test_score)

    path = f'{Path.cwd().joinpath(f"models/{dep_var}_{model_name}")}'
    if not Path(path).parent.exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(clf, f)

    ml_prediction_name = f'{dep_var}_ML'
    ml_prediction_error_name = f'{dep_var}_MLerr'
    df[ml_prediction_name] = clf.predict(X)
    df[ml_prediction_error_name] = np.abs(df[dep_var] - df[ml_prediction_name])

    excel_filename = Path.cwd().joinpath('models/ML_error_analysis.xlsx')
    ml_dfs = {}
    if Path.exists(excel_filename):
        wb = openpyxl.load_workbook(excel_filename)
        for sheet in wb.sheetnames:
            ml_dfs[sheet] = pd.read_excel(excel_filename, sheet_name=sheet)

    ml_df = df[ml_prediction_error_name].describe(
        percentiles=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99])

    writer = pd.ExcelWriter(excel_filename, engine='openpyxl')
    if Path.exists(excel_filename):
        for mldf in ml_dfs.keys():
            ml_dfs[mldf].to_excel(writer, sheet_name=mldf)

    ml_df.to_excel(writer, sheet_name=ml_prediction_error_name)
    writer.save()
    writer.close()

    fig, ax = plt.subplots()
    ax.scatter(x=indep_vars[0], y=dep_var, data=df, c='green')
    ax.scatter(x=indep_vars[0], y=ml_prediction_name, data=df, c='red')
    leg = plt.legend(loc='best')
    leg.set_draggable(state=True)

if __name__ == '__main__':
    df = pd.read_hdf(r'C:/Temp/0704.h5')
    indep_vars = ['B_NLR17_C', 'B_WC_AIR_MNFD_P', 'U_TAMB', 'U_RH', 'B_FOG_N_STAGE']
    dep_var = 'U_MWG'
    learn(df, indep_vars, dep_var, regression=True, plot=True)
