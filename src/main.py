import pandas as pd
from sklearn.linear_model import LinearRegression

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


from .model.train_model import create_listing_price_model, compare_models




model = create_listing_price_model('Training_DataSet.csv', LinearRegression())

compare_models('Training_DataSet.csv')