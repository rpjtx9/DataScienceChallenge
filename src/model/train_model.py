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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.impute import SimpleImputer

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score, explained_variance_score

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

import os

from ..data.dataset import get_listing_price_dataframe, get_vehicle_trim_dataframe

from ..helpers.helpers import data_path



def get_listing_price_feats_targets(master_df : pd.DataFrame):
    price = master_df[master_df['Dealer_Listing_Price'].notnull()]

    features = price.drop(columns='Dealer_Listing_Price')

    targets = pd.DataFrame(price['Dealer_Listing_Price'])
    features.sort_index(axis = 1, inplace = True)
    targets.sort_index(axis = 1, inplace = True)
    trim_map = 1
    return (features, targets, trim_map)

def get_vehicle_trim_feats_targets(master_df : pd.DataFrame):
        
    trim = master_df[master_df['Vehicle_Trim'].notnull()]

    trim_list = list(trim['Vehicle_Trim'].unique())

    trim_list.sort()

    trim_map = {trim_list[i] : i for i in range(len(trim_list))}

    
    features = trim.drop(columns = 'Vehicle_Trim')
    
    targets = pd.DataFrame(trim['Vehicle_Trim'])

    for item in trim_map:
        targets['Vehicle_Trim'].replace(item, trim_map[item], inplace = True)

    features.sort_index(axis = 1, inplace = True)
    targets.sort_index(axis = 1, inplace = True)
    return (features, targets, trim_map)

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

    return training_data, testing_data, imputer

def scale_values(training_data : pd.DataFrame, testing_data : pd.DataFrame, train_tgts : pd.DataFrame, test_tgts : pd.DataFrame):
    scaler = MinMaxScaler(feature_range = (0, 1))

    scaler.fit(training_data)

    scaler.transform(training_data)
    scaler.transform(testing_data)
    
    scaled_training_data = training_data
    scaled_testing_data = testing_data

    scaled_training_answers = np.array(train_tgts).reshape((-1, ))
    scaled_testing_answers = np.array(test_tgts).reshape((-1, ))

    return scaled_training_data, scaled_testing_data, scaled_training_answers, scaled_testing_answers, scaler

def impute_and_scale(train_tgts : pd.DataFrame, test_tgts : pd.DataFrame, train_feats : pd.DataFrame, test_feats : pd.DataFrame):

    training_data, testing_data, imputer = impute_missing_values(train_feats, test_feats)

    final_training_data, final_testing_data, final_training_answers, final_testing_answers, scaler = scale_values(training_data, testing_data, train_tgts, test_tgts)

    pd.DataFrame(final_testing_data).to_csv('F:/Documents/Projects/DataScienceChallenge/data/Scaled_Trained_Listing_Price.csv', index = False)


    return final_training_data, final_testing_data, final_training_answers, final_testing_answers, imputer, scaler
    

def mean_absolute_error(test_target, test_prediction):
    return np.mean(abs(test_target - test_prediction), axis = 0)

def fit_and_evaluate(model_type, training_data : pd.DataFrame, training_answers : pd.DataFrame, testing_data : pd.DataFrame, testing_answers : pd.DataFrame, VehicleTrim = True, trim_map = None, cv_test = False):

    # Train model:
    model_type.fit(training_data, training_answers)


    # Make predictions
    model_pred = model_type.predict(testing_data)
    model_mae = mean_absolute_error(testing_answers, model_pred)
    

    # if VehicleTrim:
    #     # Print Evaluation for categorical target

    #     print(classification_report(testing_answers, model_pred))
    #     conf_matrix = confusion_matrix(testing_answers, model_pred)
    #     disp = ConfusionMatrixDisplay(conf_matrix)
    #     acc = accuracy_score(testing_answers, model_pred)
    #     print(acc)
    #     disp.plot()
    #     plt.show()
    
    # else:
    #     # Print evaluation for numerical target
    #     print('Model performance on the test set: MAE = %0.4f' % model_mae)
    #     acc = explained_variance_score(testing_answers, model_pred)
    #     print(acc)
    
    if cv_test:
        cross_value(model_type, training_data, training_answers)
        



    model = model_type
    return model

def create_model(dataset_filename, model_type, VehicleTrim = True, cv_test = False):
    if VehicleTrim:
        master_df = get_vehicle_trim_dataframe(dataset_filename)
        features, targets, trim_map = get_vehicle_trim_feats_targets(master_df)
    else:
        master_df = get_listing_price_dataframe(dataset_filename)
        features, targets, trim_map = get_listing_price_feats_targets(master_df)

    train_feats, test_feats, train_tgts, test_tgts = split_data(features, targets)

    training_data, testing_data, training_answers, testing_answers, imputing_fit, scaling_fit = impute_and_scale(train_tgts, test_tgts, train_feats, test_feats)

    model = fit_and_evaluate(model_type, training_data, training_answers, testing_data, testing_answers, VehicleTrim, trim_map, cv_test)

    return model, trim_map, training_data, testing_data, training_answers, testing_answers, imputing_fit, scaling_fit





def compare_models(dataset_filename, VehicleTrim = True):

    plt.style.use('fivethirtyeight')
    figsize(15, 6)

    if VehicleTrim:
        master_df = get_vehicle_trim_dataframe(dataset_filename)
        features, targets, trim_map = get_vehicle_trim_feats_targets(master_df)

    else:
        master_df = get_listing_price_dataframe(dataset_filename)

        features, targets = get_listing_price_feats_targets(master_df)

    train_feats, test_feats, train_tgts, test_tgts = split_data(features, targets)

    training_data, testing_data, training_answers, testing_answers = impute_and_scale(train_tgts, test_tgts, train_feats, test_feats)

    model_types = [LinearRegression(), SVR(C = 1000, gamma = 0.1), RandomForestRegressor(random_state = 69), GradientBoostingRegressor(random_state = 60), KNeighborsRegressor(n_neighbors = 10)]

    mae = []
    for modeltype in model_types:
        model = fit_and_evaluate(modeltype, training_data, training_answers, testing_data, testing_answers, VehicleTrim)
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



def cross_value(model, training_data, training_answers):
    n_estimators = [10, 50, 100, 200, 300, 500, 750, 1000]
    criterion = ['gini', 'entropy', 'log_loss']
    # Maximum depth of each tree
    max_depth = [0, 2, 3, 5, 10, 15]

    # Minimum number of samples per leaf
    min_samples_leaf = [0, 1, 2, 4, 6, 8]

    # Minimum number of samples to split a node
    min_samples_split = [0, 2, 4, 6, 10]

    # Maximum number of features to consider for making splits
    max_features = ['auto', 'sqrt', 'log2', None]

    hyperparameter_grid = {'criterion': criterion,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

    random_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter_grid, cv = 4, n_iter = 25, n_jobs = -1, verbose = 1, return_train_score = True, random_state = 42)

    random_cv.fit(training_data, training_answers)

    results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)

    results.to_csv('F:/Documents/Projects/DataScienceChallenge/data/random_results.csv')

    print(random_cv.best_estimator_)

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