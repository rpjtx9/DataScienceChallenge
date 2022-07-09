import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Disable chain warnings for the following replacement function, it's working as intended and the warning is a false alarm
pd.options.mode.chained_assignment = None  # default = 'warn'
# Options for viewing the dataframe:
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_row', None)
# pd.set_option('display.max_colwidth', None)
import os
import numpy as np
# Turn off divide by zero errors
np.seterr(divide = 'ignore') # default = 'warn'

from .helpers.helpers import data_path

from .model.train_model import create_model, compare_models

from .data.dataset import clean_dataframe, get_vehicle_trim_dataframe, get_listing_price_dataframe




# Get VehTrim model. These parameters were the best ones at ~68% accuracy.

model_1, trim_map_1, training_data_1, testing_data_1, training_answers_1, testing_answers_1, imputer_1, scaler_1 = create_model('Training_DataSet.csv', RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42, max_depth = 15, min_samples_split = 10, max_features = None), VehicleTrim = True, cv_test = False)



# Get Dealer_Listing_Price model which requires vehicle trim. This parameter worked very well at ~79% accuracy

model_2, trim_map_2, training_data_2, testing_data_2, training_answers_2, testing_answers_2, imputer_2, scaler_2 = create_model('Training_DataSet.csv', LinearRegression(), VehicleTrim = False, cv_test = False)




test_column_names = list(pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'),nrows=1))

raw_dataset = pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'), usecols = [column for column in test_column_names if column not in ('VehSellerNotes', 'VehBodystyle', 'VehType', 'VehFeats', 'VehTransmission',  'SellerZip', 'SellerState', 'SellerName')])

listing_id = raw_dataset['ListingID']

raw_dataset.drop(columns = 'ListingID', inplace = True)

df = clean_dataframe(raw_dataset)

df.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Cleaned_Test_Dataset.csv', index = False)

categories = df[['SellerCity', 'SellerListSrc', 'VehColorExt', 'VehFuel',  'VehMake', 'VehModel', 'VehPriceLabel', 'IntColor1']]
categories = pd.get_dummies(categories)

test_features = pd.concat([df, categories], axis = 1)

test_features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Transformed_Test_Dataset.csv', index = False)


training_features_columns = get_vehicle_trim_dataframe('Training_DataSet.csv').columns

test_features_columns = test_features.columns


drop_columns = []
for column in test_features_columns:
    if column not in training_features_columns:
        drop_columns.append(column)

test_features.drop(columns = drop_columns, inplace = True)


for column in training_features_columns:
    if column == 'Vehicle_Trim':
        next
    elif column not in test_features_columns:
        test_features[column] = np.nan



test_features.sort_index(axis = 1, inplace = True)

test_features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Final_Transformed_Test_Dataset_Vehicle_Trim.csv', index = False)





test_data = imputer_1.transform(test_features)
scaler_1.transform(test_data)

pd.DataFrame(test_data).to_csv('F:/Documents/Projects/DataScienceChallenge/data/Scaled_Test_Vehicle_Trim.csv', index = False)


trim_model_pred = model_1.predict(test_data)

trim_model_pred = list(trim_model_pred)

for i, item in enumerate(trim_model_pred):
    for trim, value in trim_map_1.items():
        if item == value:
            trim_model_pred[i] = trim


#Start with the second model

test_column_names = list(pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'),nrows=1))

raw_dataset = pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'), usecols = [column for column in test_column_names if column not in ('VehSellerNotes', 'VehBodystyle', 'VehType', 'VehFeats', 'VehTransmission',  'SellerZip', 'SellerState', 'SellerName')])

listing_id = raw_dataset['ListingID']

raw_dataset.drop(columns = 'ListingID', inplace = True)
raw_dataset['Vehicle_Trim'] = trim_model_pred

df = clean_dataframe(raw_dataset)



categories = df[['SellerCity', 'SellerListSrc', 'VehColorExt', 'VehFuel',  'VehMake', 'VehModel', 'VehPriceLabel', 'IntColor1', 'Vehicle_Trim']]
categories = pd.get_dummies(categories)

test_features = pd.concat([df, categories], axis = 1)

test_features.sort_index(axis = 1, inplace = True)


training_features_columns = get_listing_price_dataframe('Training_DataSet.csv').columns

test_features_columns = test_features.columns


drop_columns = []
for column in test_features_columns:
    if column not in training_features_columns:
        drop_columns.append(column)

test_features.drop(columns = drop_columns, inplace = True)


for column in training_features_columns:
    if column == 'Dealer_Listing_Price':
        next
    elif column not in test_features_columns:
        test_features[column] = np.nan

test_features.sort_index(axis = 1, inplace = True)

test_features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Final_Transformed_Test_Dataset_Listing_Price.csv', index = False)

scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(training_data_2)

test_data = imputer_2.transform(test_features)

scaler.transform(test_data)

pd.DataFrame(test_data).to_csv('F:/Documents/Projects/DataScienceChallenge/data/Scaled_Test_Listing_Price.csv', index = False)

price_model_pred = model_2.predict(test_data)


answers = pd.DataFrame()
answers['ListingID'] = listing_id
answers['VehicalTrim'] = trim_model_pred
answers['Dealer_Listing_Price'] = price_model_pred

answers.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Data_Science_Challenge_Answers.csv', index = False)