import torch
import math
import pandas as pd
import os
import numpy as np
from .. helpers.helpers import root_path, data_path, file_path
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

import seaborn as sns

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_row', None)
# pd.set_option('display.max_colwidth', None)

# Get the list of all column names so we can exclude reading some rows for efficiency
column_names = list(pd.read_csv(os.path.join(data_path, "Training_Dataset.csv"),nrows=1))

# Read the file into a dataframe, exclude rows we don't want.
# Excluding seller notes because they're not trustworthy. Bodystyle and VehType because 100% of the body style is SUV. VehFeats is unreliable due to the different reporting styles between sellers
df = pd.read_csv(os.path.join(data_path, "Training_Dataset.csv"), usecols = [column for column in column_names if column not in ('VehSellerNotes', 'VehBodystyle', 'VehType', 'VehFeats')])

# Convert zip codes to strings
df['SellerZip'] = df['SellerZip'].astype(str)

history = set(df["VehHistory"].to_list())

# Pull individual elements of Vehicle History into a Set that can be added to the data frame
history = set()
for element in df['VehHistory'].iloc:
    element_list = str(element).strip("[]").replace("'", "").split(",")
    element_list = [str(string).strip() for string in element_list]
    for string in element_list:
        history.add(string)


print(history)
# for key in df:
#     print(df[key].describe(), '\n')