from cmath import nan
import spellchecker
import torch
import math
import pandas as pd
import os
import numpy as np
from .. helpers.helpers import root_path, data_path, file_path
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import re
from spellchecker import SpellChecker

import seaborn as sns

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_row', None)
# pd.set_option('display.max_colwidth', None)

def clean_sales_dataframe():
    '''
    Function takes no arguments and returns the car sales dataframe after being cleaned.

    Cleaning involved:
        Changing Zip Code to string type
        Exploding each item in Vehicle History into a new column with 1 = True and 0 = False and dropping the original Vehicle History column
        Exploding the VehDriveTrain column into new columns with 4WD and AWD. 1 = True and 0 = False. Then dropping  the original VehDriveTrain column
        Getting the number of cylinders and the cylinder size from the  VehEngine column and dropping the original VehEngine column
        Splitting the
    '''
    # Get the list of all column names so we can exclude reading some rows for efficiency
    column_names = list(pd.read_csv(os.path.join(data_path, "Training_Dataset.csv"),nrows=1))

    # Read the file into a dataframe, exclude rows we don't want.
    # Excluding seller notes because they're not trustworthy. Bodystyle and VehType because 100% of the body style is SUV. VehFeats is unreliable due to the different reporting styles between sellers
    df = pd.read_csv(os.path.join(data_path, "Training_Dataset.csv"), usecols = [column for column in column_names if column not in ('VehSellerNotes', 'VehBodystyle', 'VehType', 'VehFeats', 'VehTransmission', 'ListingID')])

    # Convert zip codes to strings
    df['SellerZip'] = df['SellerZip'].astype(str)



    # Cleaning the Vehicle History column:
    # VehHistory has multiple values in each cell, but they are standardized. We want to pull those out and create new columns for them.

    # Pull individual elements of Vehicle History into a Set that can be added to the data frame
    history = set()
    # This returns a list of the values one row at a time
    for element in df['VehHistory'].iloc:
        element_list = str(element).strip("[]").replace("'", "").split(",")
        element_list = [str(string).strip() for string in element_list]
        # This will attempt to add each value to the set
        for string in element_list:
            if string != 'nan':
                history.add(string)

    # Sort the set to set column order
    history = sorted(history)

    # Before adding values we need to change NaN to No History because np.where will always evaluate NaN as True.
    df['VehHistory'] = df['VehHistory'].replace({np.nan : 'No History'})
    # Now add new Vehicle History columns to the dataframe and populate them if the VehHistory column contains the substring, then drop the original VehHistory column since it's not needed
    for name in  history:
        df[name] = np.where(df['VehHistory'].str.contains(name, regex = False), 1, 0)
    df.drop(columns = 'VehHistory', inplace = True)



    # Cleaning the VehDriveTrain Column:

    # VehDriveTrain starts with these values:
        # 4X4
        # <null>
        # 4x4/4WD
        # 4WD
        # FWD
        # AWD
        # 4x4
        # Four Wheel Drive
        # FRONT-WHEEL DRIVE
        # All Wheel Drive
        # ALL-WHEEL DRIVE WITH LOCKING AND LIMITED-SLIP DIFFERENTIAL
        # AWD or 4x4
        # ALL-WHEEL DRIVE
        # Front Wheel Drive
        # 4x4/4-wheel drive
        # All-wheel Drive
        # Front-wheel Drive
        # 2WD
        # ALL WHEEL
        # AllWheelDrive
        # 4WD/AWD
    # We will create 2 new columns, All Wheel Drive (AWD) and 4 Wheel Drive (4WD) with 1 for True and 0 for False.

    # Start by changing NaN to "No Information" otherwise np.where will evaluate NaN as True
    df['VehDriveTrain'] = df['VehDriveTrain'].replace({np.nan : 'No Information'})
    # Then create the new columns if they contain the relevant substrings
    df['AWD'] = np.where(df['VehDriveTrain'].str.contains('AWD|All Wheel Drive|All-Wheel Drive|All Wheel|AllWheelDrive', flags=re.IGNORECASE, regex=True), 1, 0)
    df['4WD'] = np.where(df['VehDriveTrain'].str.contains('4x4|4WD|Four Wheel Drive', flags=re.IGNORECASE, regex=True), 1, 0)

    # Drop VehDriveTrain now that we're done with it

    df.drop(columns = 'VehDriveTrain', inplace = True)

    # Next clean VehEngine
    # The information needing out of VehEngine is how many cylinders it has and what size the cylinders are. I guess we can pull Hemi too.

    # Start by changing NaN to "No Information" otherwise np.where will evaluate NaN as True
    df['VehEngine'] = df['VehEngine'].replace({np.nan : 'No Information'})
    # Then create the cylinders column and populate it. 

    #Grabbing all columns that say V6, V8, V-6, etc
    df['NumCylinders'] = df['VehEngine'].str.extract(r'V.?(\d)', flags = re.IGNORECASE)


    # Then fill the remaining columns that say X Cylinders. Set datatypes to Int64 which allows for nullable integers
    df['NumCylinders'] = df['NumCylinders'].fillna(df['VehEngine'].str.extract(r'(\d).?cyl', expand = False, flags = re.IGNORECASE)).astype('Int64')

    # Next we will pull the cylinder size from VehEngine into its own column.
    df['CylinderSize'] = df['VehEngine'].str.extract(r'(\d\.\d).?L', flags = re.IGNORECASE)

    # Finally we will check for a HEMI
    df['HEMI'] = np.where(df['VehEngine'].str.contains('HEMI', regex = False, flags = re.IGNORECASE), 1, 0)

    # And then drop the old VehEngine column
    df.drop(columns = 'VehEngine', inplace = True)


    # Cleaning the interior color column:
    
    # Extract multiple interior colors into IntColor1, make all lowercase
    df['IntColor1'] = df['VehColorInt'].str.extract(r'(\w+\s?\w+?\s?\w+\s?\w+\s?).*?w?/?', flags = re.IGNORECASE, expand = False).str.lower()

    # The regex occasionally pulled W's and the word leather so remove and clean up whitespace:
    df['IntColor1'] = df['IntColor1'].str.replace(' w', '').str.strip()
    df['IntColor1'] = df['IntColor1'].str.replace(' leather', '').str.strip()
    df['IntColor1'] = df['IntColor1'].str.replace(' cloth', '').str.strip()


    df[['IntColor1', 'IntColor2']] = df['IntColor1'].fillna('N/A')

    # Disable chain warnings for the following replacement function, it's working as intended and the warning is a false alarm
    pd.options.mode.chained_assignment = None  # default='warn'

    # Run a spellcheck to catch basic misspellings. Adding already found value to a dictionary to skip running spellcheck on duplicates; this dramatically improves efficiency
    colors = {}
    for i, value in enumerate(df['IntColor1']):
        if value in colors:
            df['IntColor1'][i] = colors[value]
        else:
            words = value.split(' ')
            for word in words:
                word.replace(word, SpellChecker(distance=1).correction(word))
            spellchecked_value = ' '.join(words)
            colors[value] = spellchecked_value
            df['IntColor1'][i] = colors[value]

    # Then map similar colors together. Adding values such as cirrus, maple, frost and sahara in because they are very common colors for these cars. Anything else is too small of a sample and will be marked other.
    color_map = {
        'black' : 'black',
        'beige' : 'beige',
        'frost' : 'beige',
        'sahara' : 'beige',
        'red' : 'red',
        'blue' : 'blue',
        'brown' : 'brown',
        'maple' : 'brown',
        'tan' : 'brown',
        'grey' : 'grey',
        'gray' : 'grey',
        'cirrus': 'cirrus'
    }
    for i, value in enumerate(df['IntColor1']):
        mapped = 0
        for key in color_map:
            if key in value:
                df['IntColor1'][i] = color_map[key]
                mapped = 1
        if not mapped:    
                df['IntColor1'][i] = 'other'



    
    print(df['IntColor2'].unique())
    print(df['IntColor2'].value_counts())
    # for key in df:
    #     print(df[key].describe(), '\n')



clean_sales_dataframe()