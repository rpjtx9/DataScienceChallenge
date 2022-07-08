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

from .helpers.helpers import data_path

from .data.dataset import get_listing_price_dataframe

from .model.train_model import get_baseline, get_listing_price_feats_targets, split_data, impute_missing_values, scale_values, impute_and_scale, mean_absolute_error, fit_and_evaluate, create_listing_price_model




create_listing_price_model('Training_DataSet.csv')