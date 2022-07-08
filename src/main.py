import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

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


from .model.train_model import create_model, compare_models


# Best model so far for VehicleTrim

model, trim_map, training_data, testing_data, training_answers, testing_answers = create_model('Training_DataSet.csv', RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42, max_depth = 15, min_samples_split = 10, max_features = None), VehicleTrim = True, cv_test = False)

model_pred = model_pred = model.predict(testing_data)

model_pred = list(model_pred)

for i, item in enumerate(model_pred):
    for trim, value in trim_map.items():
        if item == value:
            model_pred[i] = trim

print(testing_data)

# # Best model for Dealer_Listing_Price

# model, trim_map, training_data, testing_data, training_answers, testing_answers = create_model('Training_DataSet.csv', LinearRegression(), VehicleTrim = False, cv_test = False)

# # compare_models('Training_DataSet.csv', VehicleTrim = True)