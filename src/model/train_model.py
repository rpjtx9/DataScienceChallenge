import pandas as pd
import numpy as np
from torch import grid_sampler

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
# Set default font size
plt.rcParams['font.size'] = 20

from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Scaling values
from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

import os

from ..data.dataset import get_sales_dataframe

from ..helpers.helpers import data_path


master_df = pd.read_csv(os.path.join(data_path, "Generalized_Dataset.csv"))

price = master_df[master_df['Dealer_Listing_Price'].notnull()]

features = price.drop(columns='Dealer_Listing_Price')

targets = pd.DataFrame(price['Dealer_Listing_Price'])

train_feats, test_feats, train_tgts, test_tgts = train_test_split(features, targets, test_size = 0.3, random_state = 42)

def get_baseline():

    def mean_absolute_error(test_target, test_prediction):
        return np.mean(abs(test_target - test_prediction), axis = 0)

    baseline = np.median(train_tgts)

    print('The baseline guess is a score of %0.2f' % baseline)
    print("Baseline Performance on the test set: MAE = %0.4f" % mean_absolute_error(test_tgts, baseline))
    return baseline



imputer = SimpleImputer(strategy = 'median')

imputer.fit(train_feats)

training_data = imputer.transform(train_feats)
testing_data = imputer.transform(test_feats)

# print(f'Missing values in training features: {np.sum(np.isnan(training_data))}')
# print(f'Missing values in testing features: {np.sum(np.isnan(testing_data))}')

# print(np.where(~np.isfinite(training_data)))
# print(np.where(~np.isfinite(testing_data)))

scaler = MinMaxScaler(feature_range = (0, 1))

scaler.fit(training_data)

scaler.transform(training_data)
scaler.transform(testing_data)

training_answers = np.array(train_tgts).reshape((-1, ))
testing_answers = np.array(test_tgts).reshape((-1, ))

def mean_absolute_error(test_target, test_prediction):
    return np.mean(abs(test_target - test_prediction), axis = 0)

def fit_and_evaluate(model):

    # Train model:
    model.fit(training_data, training_answers)

    # Make predictions
    model_pred = model.predict(testing_data)
    model_mae = mean_absolute_error(testing_answers, model_pred)

    # Return performance metric
    return model_mae

# get_baseline()

# lr = LinearRegression()

# lr_mae = fit_and_evaluate(lr)

# print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)

# svm = SVR(C = 1000, gamma = 0.1)
# svm_mae = fit_and_evaluate(svm)
# print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)

# random_forest = RandomForestRegressor(random_state = 69)
# random_forest_mae = fit_and_evaluate(random_forest)
# print('Random Forest Performance on the test set: MAE = %0.4f' % random_forest_mae)

# gradient_boosted = GradientBoostingRegressor(random_state=60)
# gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

# print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)

# knn = KNeighborsRegressor(n_neighbors=10)
# knn_mae = fit_and_evaluate(knn)

# print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)

# plt.style.use('fivethirtyeight')
# figsize(15, 6)

# # Dataframe to hold the results
# model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',
#                                            'Random Forest', 'Gradient Boosted',
#                                             'K-Nearest Neighbors'],
#                                  'M.A.E.': [lr_mae, svm_mae, random_forest_mae, 
#                                          gradient_boosted_mae, knn_mae]})

# # Horizontal bar chart of test mae
# model_comparison.sort_values('M.A.E.', ascending = False).plot(x = 'model', y = 'M.A.E.', kind = 'barh',
#                                                            color = 'red', edgecolor = 'black')

# Plot formatting
# plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)
# plt.title('Model Comparison on Test Mean Absolute Error', size = 20);

# plt.tight_layout()
# plt.show()

model = LinearRegression()

model.fit(training_data, training_answers)

prediction = model.predict(testing_data)

print('Model performance on the test set: MAE = %0.4f' % mean_absolute_error(testing_answers, prediction))


figsize(8, 8)

# sns.kdeplot(prediction, label = "Predictions")
# sns.kdeplot(testing_answers, label = "Values")

# plt.xlabel('Dealer Listing Price'); plt.ylabel('Density')
# plt.title('Test Values and Predictions')



# Calculate the residuals 
# residuals = prediction - testing_answers

# Plot the residuals in a histogram
# plt.hist(residuals, color = 'red', bins = 20,
#          edgecolor = 'black')
# plt.xlabel('Error'); plt.ylabel('Count')
# plt.title('Distribution of Residuals');



# plt.legend()
# plt.show()

feature_results = pd.DataFrame({'feature': list(train_feats.columns),'importance' : model.coef_})

# Show the top 10 most important
feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)

print(feature_results.head(25))