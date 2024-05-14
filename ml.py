import time
import pickle
import pathlib
import openpyxl
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from typeguard import typechecked
from keras.optimizers import Adam
from typing import Optional, Union, List
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from tensorflow.keras.layers import Input, Dense, concatenate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor

sns.set()
matplotlib.use('Qt5Agg', force=True)


class FeatureSelector:
    model_names = ['RidgeClassifier', 'Lasso', 'LR', 'Tree', 'RF', 'Gradient Boost']
    param_grid_ridge = [{'solver': ['svd', 'cholesky', 'lsqr'], 'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}]
    param_grid_lasso = [{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'fit_intercept': [True, False]}]
    param_grid_knn = [{}, {'n_neighbors': [1, 2, 3, 4]}]
    param_grid_lr = [{'fit_intercept': [True, False]}]
    param_grid_tree = [{'random_state': [18]}, {'max_depth': [2, 3], 'min_samples_split': [3, 5]}]
    param_grid_rf = [{'random_state': [18]},
                     {'n_estimators': [10, 50], 'max_features': [0.2, 0.3], 'bootstrap': [True]}]
    param_grid_gb = [{'random_state': [18]}, {'n_estimators': [10, 50], 'max_features': [0.2, 0.3]}]

    classifiers = ([(RidgeClassifier(), model_names[0], param_grid_ridge),
                    (LogisticRegression(), model_names[1], param_grid_lr),
                    (DecisionTreeClassifier(), model_names[2], param_grid_tree),
                    (RandomForestClassifier(), model_names[3], param_grid_rf),
                    (GradientBoostingClassifier(), model_names[4], param_grid_gb)])

    regressors = ([(Ridge(), model_names[0], param_grid_ridge),
                   (Lasso(), model_names[1], param_grid_lasso),
                   (LinearRegression(), model_names[2], param_grid_lr),
                   (DecisionTreeRegressor(), model_names[3], param_grid_tree),
                   (RandomForestRegressor(), model_names[4], param_grid_rf),
                   (GradientBoostingRegressor(), model_names[5], param_grid_gb)])

    @typechecked
    def __init__(self, data: pd.DataFrame, dependent_variables: list):
        self.data = pd.DataFrame(data)
        self.selected_data = pd.DataFrame()
        self.dependent_variables = dependent_variables

    @typechecked
    def plot_learning_curve(self, estimator, title, X, y, results, ylim=None, cv=None, test_score=None,
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

    @typechecked
    def learn(self, regression: Optional[bool] = True, plot: Optional[bool] = False):
        learning_time = time.time()
        learning_rows = self.data.shape[0]
        X = self.data.drop(self.dependent_variables, axis=1).values
        y = self.data[self.dependent_variables].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=18)
        scores = []
        for i in range(len(self.model_names)):
            if regression:
                estimator = self.regressors[i][0]
                param_grid = self.regressors[i][2]
            else:
                estimator = self.classifiers[i][0]
                param_grid = self.classifiers[i][2]

            title = self.model_names[i]
            model = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='r2')
            model.fit(X_train, y_train)
            results = model.predict(X_test)
            # accuracies = [accuracy_score(y_test[:, i], results[:, i]) for i in range(len(self.dependent_variables))]
            estimator = model.best_estimator_
            scores.append((model.best_score_, estimator))
            if plot:
                self.plot_learning_curve(estimator=estimator, title=title, X=X, y=y, cv=None, n_jobs=4, results=results)

        max_score = -999999
        model_name = self.model_names[0]
        best_estimator = scores[0][1]
        for score in range(len(scores)):
            if scores[score][0] > max_score:
                max_score = scores[score][0]
                best_estimator = scores[score][1]
                model_name = self.model_names[score]

        feature_selector_model = best_estimator
        feature_selector_model.fit(X, y)
        selected = SelectFromModel(feature_selector_model, prefit=True)
        X_train_selected = selected.transform(X_train)
        X_test_selected = selected.transform(X_test)
        model = best_estimator
        X_selected = selected.transform(X)
        results = model.fit(X_train_selected, y_train)
        print(f'Machine Learning took {time.time() - learning_time} seconds for {learning_rows} rows of data')
        test_score = results.score(X_test_selected, y_test)
        self.plot_learning_curve(estimator=best_estimator, title=model_name, X=X, y=y, cv=None, n_jobs=4,
                                 results=results, grid_search=False, test_score=test_score)

        for dependent_variable in self.dependent_variables:
            ml_prediction_name = f'{dependent_variable}_ML'
            ml_prediction_error_name = f'{dependent_variable}_MLerr'
            self.data[ml_prediction_name] = model.predict(X_selected)
            self.data[ml_prediction_error_name] = np.abs(self.data[dependent_variable] - self.data[ml_prediction_name])

            fig, ax = plt.subplots()
            ax.scatter(x=self.data[self.data.drop(dependent_variable, axis=1).columns[0]],
                       y=self.data[dependent_variable],
                       c='green')
            ax.scatter(x=self.data[self.data.drop(dependent_variable, axis=1).columns[0]],
                       y=self.data[ml_prediction_name],
                       c='red')
            leg = plt.legend(loc='best')
            leg.set_draggable(state=True)

        for col in [*self.data.columns]:
            if col in self.dependent_variables or col in [*pd.DataFrame(X_selected).columns] or (
                    isinstance(col, (str,)) and 'ML' in col):
                self.selected_data[col] = self.data[col]

    @typechecked
    def amram_intelligence(self, output_sizes: List[int]):
        input_layer = Input(shape=(self.data.drop(columns=self.dependent_variables).shape[1],))

        # Define shared layers
        first_shared_layer = Dense(128, activation='relu', kernel_initializer='he_normal')(input_layer)
        second_shared_layer = Dense(64, activation='relu')(first_shared_layer)
        third_shared_layer = Dense(32, activation='relu')(second_shared_layer)

        # Define multiple output layers
        outputs = {}
        for idx, output in enumerate(output_sizes):
            if idx == 0:
                activation = 'sigmoid'
            elif idx == 1:
                activation = 'softmax'
            else:
                activation = 'linear'
            outputs[self.dependent_variables[idx]] = Dense(output, activation=activation, name=f'output{idx + 1}')(
                third_shared_layer)

        # Create the multi-output model
        model = Model(inputs=input_layer, outputs=[outputs[col] for col in self.dependent_variables])

        # Compile the model
        model.compile(optimizer=Adam(lr=0.001),
                      loss={f'output1': 'binary_crossentropy',
                            f'output2': 'sparse_categorical_crossentropy',
                            f'output3': 'mean_squared_error'},
                      metrics=['accuracy', 'mse', 'mae'])

        # Prepare the data
        X = self.data.drop(self.dependent_variables, axis=1)
        y = self.data[self.dependent_variables]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.18, random_state=18)

        output_train = {}
        output_test = {}
        for idx, col in enumerate(self.dependent_variables):
            output_train[f'output{idx + 1}'] = y_train.loc[:, col]
            output_test[f'output{idx + 1}'] = y_test.loc[:, col]

        # Train the model
        history = model.fit(X_train, output_train,
                            validation_data=(X_test, output_test), epochs=50,
                            batch_size=32)

        # Predict multiple outputs
        predictions = model.predict(X_test)

        # Plot learning curves
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        for idx, col in enumerate(self.dependent_variables):
            if idx == 0 or idx == 1:
                metric = 'accuracy'
            else:
                metric = 'mse'
            ax[idx].set_title(f'{col}')
            ax[idx].plot(history.history[f'output{idx + 1}_{metric}'], label=f'{col}_train')
            ax[idx].plot(history.history[f'val_output{idx + 1}_{metric}'], label=f'{col}_validation')
            ax[idx].legend(loc='best')
            ax[idx].set_xlabel('Epochs')
            ax[idx].set_ylabel(metric)

        plt.show()


def select_features():
    # Load and process the breast cancer dataset
    breast_cancer = load_breast_cancer()
    breast_cancer_data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    breast_cancer_target = pd.DataFrame(breast_cancer.target, columns=['breast_cancer_target'])

    # Load and process the iris dataset
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_target = pd.DataFrame(iris.target, columns=['iris_target'])

    # Load and process the diabetes dataset
    diabetes = load_diabetes()
    diabetes_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_target = pd.DataFrame(diabetes.target, columns=['diabetes_target'])

    # Standardize datasets
    scaler = StandardScaler()
    breast_cancer_data = pd.DataFrame(scaler.fit_transform(breast_cancer_data), columns=breast_cancer.feature_names)
    iris_data = pd.DataFrame(scaler.fit_transform(iris_data), columns=iris.feature_names)
    diabetes_data = pd.DataFrame(scaler.fit_transform(diabetes_data), columns=diabetes.feature_names)

    # Stack the input features horizontally
    max_rows = max(breast_cancer_data.shape[0], iris_data.shape[0], diabetes_data.shape[0])
    combined_data = pd.concat([
        breast_cancer_data.reindex(np.arange(max_rows), method='pad'),
        iris_data.reindex(np.arange(max_rows), method='pad'),
        diabetes_data.reindex(np.arange(max_rows), method='pad')
    ], axis=1)

    # Stack the targets horizontally
    combined_targets = pd.concat([
        breast_cancer_target.reindex(np.arange(max_rows), method='pad'),
        iris_target.reindex(np.arange(max_rows), method='pad'),
        diabetes_target.reindex(np.arange(max_rows), method='pad')
    ], axis=1)

    data = pd.concat([combined_data, combined_targets], axis=1)
    dependent_variables = ['breast_cancer_target', 'iris_target', 'diabetes_target']
    selected_features = FeatureSelector(data=data, dependent_variables=dependent_variables)
    # selected_features.learn()
    selected_features.amram_intelligence(output_sizes=[1, 3, 2])
    return selected_features


if __name__ == '__main__':
    important_features = select_features()
