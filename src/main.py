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




# Get the VehTrim model. These parameters were the best ones at ~68% accuracy.

model_1, trim_map_1, training_data_1, testing_data_1, training_answers_1, testing_answers_1, imputer_1, scaler_1 = create_model('Training_DataSet.csv', RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42, max_depth = 15, min_samples_split = 10, max_features = None), VehicleTrim = True, cv_test = False)


# Get Dealer_Listing_Price model which requires vehicle trim. This parameter worked very well at ~79% accuracy

model_2, trim_map_2, training_data_2, testing_data_2, training_answers_2, testing_answers_2, imputer_2, scaler_2 = create_model('Training_DataSet.csv', LinearRegression(), VehicleTrim = False, cv_test = False)


# We need to clean up the test dataset so that it matches the format of the training datasets. 
# Unfortunately this was an oversight on my part. I did not structure my functions to work different for the test dataset. 
# Should be refactored into easier-to-read and more efficient functions, but, I'm out of time.

# Pull the column names so the same columns can be excluded as were in the training set
test_column_names = list(pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'),nrows=1))
raw_dataset = pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'), usecols = [column for column in test_column_names if column not in ('VehSellerNotes', 'VehBodystyle', 'VehType', 'VehFeats', 'VehTransmission',  'SellerZip', 'SellerState', 'SellerName')])

# Save the ListingID for the final .csv and drop the column
listing_id = raw_dataset['ListingID']
raw_dataset.drop(columns = 'ListingID', inplace = True)

# Clean function should work as before. Optional write to csv
df = clean_dataframe(raw_dataset)
df.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Cleaned_Test_Dataset.csv', index = False)

# Create the dummy categories and concat with original
categories = df[['SellerCity', 'SellerListSrc', 'VehColorExt', 'VehFuel',  'VehMake', 'VehModel', 'VehPriceLabel', 'IntColor1']]
categories = pd.get_dummies(categories)
test_features = pd.concat([df, categories], axis = 1)

# Another optional write
test_features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Transformed_Test_Dataset.csv', index = False)

# We need to drop any columns that don't exist due to differences in the training and testing data set. Primarily with categories like city.

# Get the list of training and test columns
training_features_columns = get_vehicle_trim_dataframe('Training_DataSet.csv').columns
test_features_columns = test_features.columns

# Drop any from the test set that don't exist in the training set.
drop_columns = []
for column in test_features_columns:
    if column not in training_features_columns:
        drop_columns.append(column)

test_features.drop(columns = drop_columns, inplace = True)

# Then add null values for any columns that don't exist in the test set that DO exist in the training set
for column in training_features_columns:
    if column == 'Vehicle_Trim':
        next
    elif column not in test_features_columns:
        test_features[column] = np.nan

# Very important to sort before imputing/scaling
test_features.sort_index(axis = 1, inplace = True)

# Optional write to csv for documentation
test_features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Final_Transformed_Test_Dataset_Vehicle_Trim.csv', index = False)



# Impute and scale.
test_data = imputer_1.transform(test_features)
scaler_1.transform(test_data)

# Optional write to check scaling, I had a lot of trouble with this. Can't assign scaler.transform for some reason or it doesn't work? But it did on training data? I don't know.
pd.DataFrame(test_data).to_csv('F:/Documents/Projects/DataScienceChallenge/data/Scaled_Test_Vehicle_Trim.csv', index = False)

# Make predictions and cast to a list so we can index through and re-map the numbers back to their appropriate Trim
trim_model_pred = model_1.predict(test_data)
trim_model_pred = list(trim_model_pred)

for i, item in enumerate(trim_model_pred):
    for trim, value in trim_map_1.items():
        if item == value:
            trim_model_pred[i] = trim


# Repeat for the 2nd model. 
# Definitely needs to  be refactored so all this code doesn't repeat and share so many variable names. This is unstable code.

# Pull the column names so the same columns can be excluded as were in the training set. 
test_column_names = list(pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'),nrows=1))
raw_dataset = pd.read_csv(os.path.join(data_path, 'Test_Dataset.csv'), usecols = [column for column in test_column_names if column not in ('VehSellerNotes', 'VehBodystyle', 'VehType', 'VehFeats', 'VehTransmission',  'SellerZip', 'SellerState', 'SellerName')])

# Save the ListingID for the final .csv and drop the column
listing_id = raw_dataset['ListingID']
raw_dataset.drop(columns = 'ListingID', inplace = True)

# Add the predictions from the first model into the dataset
raw_dataset['Vehicle_Trim'] = trim_model_pred

# Cleaning should work fine
df = clean_dataframe(raw_dataset)

# Get dummy categories
categories = df[['SellerCity', 'SellerListSrc', 'VehColorExt', 'VehFuel',  'VehMake', 'VehModel', 'VehPriceLabel', 'IntColor1', 'Vehicle_Trim']]
categories = pd.get_dummies(categories)

# Concat the dummies and then SORT!
test_features = pd.concat([df, categories], axis = 1)

test_features.sort_index(axis = 1, inplace = True)

# We need to drop any columns that don't exist due to differences in the training and testing data set. Primarily with categories like city.

# Get the list of training and test columns
training_features_columns = get_listing_price_dataframe('Training_DataSet.csv').columns
test_features_columns = test_features.columns

# Drop any from the test set that don't exist in the training set.
drop_columns = []
for column in test_features_columns:
    if column not in training_features_columns:
        drop_columns.append(column)

test_features.drop(columns = drop_columns, inplace = True)

# Then add null values for any columns that don't exist in the test set that DO exist in the training set
for column in training_features_columns:
    if column == 'Dealer_Listing_Price':
        next
    elif column not in test_features_columns:
        test_features[column] = np.nan

# Need to sort again
test_features.sort_index(axis = 1, inplace = True)

test_features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Final_Transformed_Test_Dataset_Listing_Price.csv', index = False)

# Redefined this scaler because scaling was giving me all sorts of trouble. It may not be necessary but I'm leaving it for now.
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(training_data_2)

test_data = imputer_2.transform(test_features)

scaler.transform(test_data)

pd.DataFrame(test_data).to_csv('F:/Documents/Projects/DataScienceChallenge/data/Scaled_Test_Listing_Price.csv', index = False)

# Save predictions
price_model_pred = model_2.predict(test_data)

# Create answers csv file
answers = pd.DataFrame()
answers['ListingID'] = listing_id
answers['VehicalTrim'] = trim_model_pred
answers['Dealer_Listing_Price'] = price_model_pred

answers.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Data_Science_Challenge_Answers.csv', index = False)