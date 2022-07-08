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

from ..data.dataset import get_listing_price_dataframe

from ..helpers.helpers import data_path




def get_listing_price_feats_targets(master_df : pd.DataFrame):
    price = master_df[master_df['Dealer_Listing_Price'].notnull()]

    features = price.drop(columns='Dealer_Listing_Price')

    targets = pd.DataFrame(price['Dealer_Listing_Price'])

    return (features, targets)

def split_data(features : pd.DataFrame, targets : pd.DataFrame):

    train_feats, test_feats, train_tgts, test_tgts = train_test_split(features, targets, test_size = 0.3, random_state = 42)

    return train_feats, test_feats, train_tgts, test_tgts

def get_baseline(train_tgts):

    baseline = np.median(train_tgts)

    print('The baseline guess is a score of %0.2f' % baseline)

    return baseline


def impute_missing_values(train_feats : pd.DataFrame, test_feats : pd.DataFrame):
    imputer = SimpleImputer(strategy = 'median')

    imputer.fit(train_feats)

    training_data = imputer.transform(train_feats)
    testing_data = imputer.transform(test_feats)

    # print(f'Missing values in training features: {np.sum(np.isnan(training_data))}')
    # print(f'Missing values in testing features: {np.sum(np.isnan(testing_data))}')

    # print(np.where(~np.isfinite(training_data)))
    # print(np.where(~np.isfinite(testing_data)))

    return training_data, testing_data

def scale_values(training_data : pd.DataFrame, testing_data : pd.DataFrame, train_tgts : pd.DataFrame, test_tgts : pd.DataFrame):
    scaler = MinMaxScaler(feature_range = (0, 1))

    scaler.fit(training_data)

    scaler.transform(training_data)
    scaler.transform(testing_data)
    
    scaled_training_data = training_data
    scaled_testing_data = testing_data

    scaled_training_answers = np.array(train_tgts).reshape((-1, ))
    scaled_testing_answers = np.array(test_tgts).reshape((-1, ))

    return scaled_training_data, scaled_testing_data, scaled_training_answers, scaled_testing_answers

def impute_and_scale(train_tgts : pd.DataFrame, test_tgts : pd.DataFrame, train_feats : pd.DataFrame, test_feats : pd.DataFrame):

    training_data, testing_data = impute_missing_values(train_feats, test_feats)

    final_training_data, final_testing_data, final_training_answers, final_testing_answers = scale_values(training_data, testing_data, train_tgts, test_tgts)

    return final_training_data, final_testing_data, final_training_answers, final_testing_answers
    

def mean_absolute_error(test_target, test_prediction):
    return np.mean(abs(test_target - test_prediction), axis = 0)

def fit_and_evaluate(model_type, training_data : pd.DataFrame, training_answers : pd.DataFrame, testing_data : pd.DataFrame, testing_answers : pd.DataFrame):

    # Train model:
    model_type.fit(training_data, training_answers)


    # Make predictions
    model_pred = model_type.predict(testing_data)
    model_mae = mean_absolute_error(testing_answers, model_pred)

    # Print evaluation
    print('Model performance on the test set: MAE = %0.4f' % model_mae)
    # Return performance metric

    model = model_type
    return model

def create_listing_price_model(dataset_filename, model_type):


    master_df = get_listing_price_dataframe(dataset_filename)

    features, targets = get_listing_price_feats_targets(master_df)

    train_feats, test_feats, train_tgts, test_tgts = split_data(features, targets)

    training_data, testing_data, training_answers, testing_answers = impute_and_scale(train_tgts, test_tgts, train_feats, test_feats)

    model = fit_and_evaluate(model_type, training_data, training_answers, testing_data, testing_answers)

    return model





def compare_models(dataset_filename):

    plt.style.use('fivethirtyeight')
    figsize(15, 6)

    master_df = get_listing_price_dataframe(dataset_filename)

    features, targets = get_listing_price_feats_targets(master_df)

    train_feats, test_feats, train_tgts, test_tgts = split_data(features, targets)

    training_data, testing_data, training_answers, testing_answers = impute_and_scale(train_tgts, test_tgts, train_feats, test_feats)

    model_types = [LinearRegression(), SVR(C = 1000, gamma = 0.1), RandomForestRegressor(random_state = 69), GradientBoostingRegressor(random_state = 60), KNeighborsRegressor(n_neighbors = 10)]

    mae = []
    for modeltype in model_types:
        model = fit_and_evaluate(modeltype, training_data, training_answers, testing_data, testing_answers)
        model_pred = model.predict(testing_data)
        model_mae = mean_absolute_error(testing_answers, model_pred)
        mae.append(model_mae)

    model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',
                                            'Random Forest', 'Gradient Boosted',
                                                'K-Nearest Neighbors'],
                                    'M.A.E.': mae})

    # Horizontal bar chart of test mae
    model_comparison.sort_values('M.A.E.', ascending = False).plot(x = 'model', y = 'M.A.E.', kind = 'barh',
                                                            color = 'red', edgecolor = 'black')

    # Plot formatting

    plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)
    plt.title('Model Comparison on Test Mean Absolute Error', size = 20);

    plt.tight_layout()
    plt.show()

# model = LinearRegression()

# model.fit(training_data, training_answers)

# prediction = model.predict(testing_data)

# print('Model performance on the test set: MAE = %0.4f' % mean_absolute_error(testing_answers, prediction))


# figsize(8, 8)

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

# feature_results = pd.DataFrame({'feature': list(train_feats.columns),'importance' : model.coef_})

# # Show the top 10 most important
# feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)

# print(feature_results.head(25))