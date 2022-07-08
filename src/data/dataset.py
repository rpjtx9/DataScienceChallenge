import spellchecker
import pandas as pd

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
from .. helpers.helpers import root_path, data_path, file_path
import re
from spellchecker import SpellChecker





def get_sales_dataframe():
    '''
    Function takes no arguments and returns the car sales dataframe after being cleaned.

    Cleaning involves:
        Changing Zip Code to string type
        Exploding each item in Vehicle History into a new column with 1 = True and 0 = False and dropping the original Vehicle History column
        Exploding the VehDriveTrain column into new columns with 4WD and AWD. 1 = True and 0 = False. Then dropping  the original VehDriveTrain column
        Getting the number of cylinders and the cylinder size from the  VehEngine column and dropping the original VehEngine column
        Homogenizing Interior colors
        Determining if there  is a trycoat and homogenizing exterior colors, dropping any that have less than 5
    '''
    # Get the list of all column names so we can exclude reading some rows for efficiency
    column_names = list(pd.read_csv(os.path.join(data_path, "Training_Dataset.csv"),nrows=1))

    # Read the file into a dataframe, exclude rows we don't want.
    # Excluding:
    # Seller notes because there's no standard information to pull out, it's just noise
    # Bodystyle and VehType because they are zero variance variables
    # VehFeats is unreliable due to the different reporting styles between sellers. 
    # ListingID doesn't contribute.
    # SellerZip, SellerCity, and SellerState will all correlate too closely with one another. SellerZip seems too specific and has too many categories, SellerState will lose too much information. Going to go with keeping only SellerCity.
    # SellerName and SellerRating  will also all correlate too closely with one another. Going to keep SellerRating since it's a continuous variable and should be more telling overall.
    df = pd.read_csv(os.path.join(data_path, "Training_Dataset.csv"), usecols = [column for column in column_names if column not in ('VehSellerNotes', 'VehBodystyle', 'VehType', 'VehFeats', 'VehTransmission', 'ListingID', 'SellerZip', 'SellerState', 'SellerName')])

    def clean_SellerCity():
        df['SellerCity'] = df['SellerCity'].fillna('not specified')
        frequency = df['SellerCity'].value_counts(normalize= True)
        # Get rid of all cities that don't at least appear 1% of the time in the dataset
        for city in df['SellerCity'].values:
            if frequency[city] < 0.01:
                df['SellerCity'] = df['SellerCity'].replace(city, 'other')


    def clean_VehHistory():
        '''
        Pulls the unique categories out of the VehHistory column and creates new columns for each
        '''
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
            df[name] = np.where(df['VehHistory'].str.contains(name, regex = False, flags = re.IGNORECASE), 1, 0)
        df.drop(columns = 'VehHistory', inplace = True)


    def clean_VehDriveTrain():
        '''
        Converts all drive train values into All Wheel Drive (AWD) and 4 Wheel Drive (4WD) columns
        '''

        # Start by changing NaN to "No Information" otherwise np.where will evaluate NaN as True
        df['VehDriveTrain'] = df['VehDriveTrain'].replace({np.nan : 'No Information'})
        # Then create the new columns if they contain the relevant substrings
        df['AWD'] = np.where(df['VehDriveTrain'].str.contains('AWD|All Wheel Drive|All-Wheel Drive|All Wheel|AllWheelDrive', flags=re.IGNORECASE, regex=True), 1, 0)
        df['4WD'] = np.where(df['VehDriveTrain'].str.contains('4x4|4WD|Four Wheel Drive', flags=re.IGNORECASE, regex=True), 1, 0)

        # Drop VehDriveTrain now that we're done with it

        df.drop(columns = 'VehDriveTrain', inplace = True)


    def clean_VehEngine():
        '''
        Cleans the VehEngine column by pulling out the number of cylinders, the cylinder size, and whether or not it is a HEMI
        '''
        # The information needing out of VehEngine is how many cylinders it has and what size the cylinders are. I guess we can pull Hemi too.

        # Start by changing NaN to "No Information" otherwise np.where will evaluate NaN as True
        df['VehEngine'] = df['VehEngine'].replace({np.nan : 'No Information'})
        # Then create the cylinders column and populate it. 

        #Grabbing all columns that say V6, V8, V-6, etc
        df['NumCylinders'] = df['VehEngine'].str.extract(r'V.?(\d)', flags = re.IGNORECASE)


        # Then fill the remaining columns that say X Cylinders. Set datatypes to Int64 which allows for nullable integers
        df['NumCylinders'] = df['NumCylinders'].fillna(df['VehEngine'].str.extract(r'(\d).?cyl', expand = False, flags = re.IGNORECASE)).astype('Int64')

        # Next we will pull the cylinder size from VehEngine into its own column.
        df['CylinderSize'] = df['VehEngine'].str.extract(r'(\d\.\d).?L', flags = re.IGNORECASE, expand = False)
        # Convert to float64. 
        df['CylinderSize'] = df['CylinderSize'].astype(np.float64)

        # Finally we will check for a HEMI
        df['HEMI'] = np.where(df['VehEngine'].str.contains('HEMI', regex = False, flags = re.IGNORECASE), 1, 0)

        # And then drop the old VehEngine column
        df.drop(columns = 'VehEngine', inplace = True)


    def clean_VehColorInt():
        '''
        Cleans the VehColorInt column by pulling out the main interior color and simplfying it into black, beige, red, blue, brown, grey, cirrus, and other
        '''
        
        # Extract multiple interior colors into IntColor1, make all lowercase
        df['IntColor1'] = df['VehColorInt'].str.extract(r'(\w+\s?\w+?\s?\w+\s?\w+\s?).*?w?/?', flags = re.IGNORECASE, expand = False).str.lower()

        # The regex occasionally pulled W's and the word leather so remove and clean up whitespace:
        df['IntColor1'] = df['IntColor1'].str.replace(' w', '').str.strip()
        df['IntColor1'] = df['IntColor1'].str.replace(' leather', '').str.strip()
        df['IntColor1'] = df['IntColor1'].str.replace(' cloth', '').str.strip()


        df['IntColor1'] = df['IntColor1'].fillna('N/A')



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
        # This results more others than I'd prefer (in the test data set, 873 others out of ~6300 datapoints) but I believe overall it's the right call
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
            'cirrus': 'cirrus',
            'N/A' : 'N/A'    
            }
        for i, value in enumerate(df['IntColor1']):
            mapped = 0
            for key in color_map:
                if key in value:
                    df['IntColor1'][i] = color_map[key]
                    mapped = 1
            if not mapped:    
                    df['IntColor1'][i] = 'other'

        # Drop the VehColorInt column now that we are done with it
        df.drop(columns = 'VehColorInt', inplace = True)


    def clean_VehColorExt():
        '''
        Clean VehColorExt column
        '''


        # First we want to determine if there is a tricoat or metallic paint utilized as those are more expensive colors

        df['Tricoat'] = np.where(df['VehColorExt'].str.contains('tri.?co|pearl|3.?co|crystal', regex = True, flags = re.IGNORECASE), 1, 0)
        df['Metallic'] = np.where(df['VehColorExt'].str.contains('metallic', regex = True, flags = re.IGNORECASE), 1, 0)

        # Since we've pulled this information out we can now drop these words from the column and strip any leftover whitespace
        df['VehColorExt'] = df['VehColorExt'].str.replace(r'(pearl\b|\w+?.?\s?coat\b|metallic|crystal|coat)', '', regex = True, flags = re.IGNORECASE).str.strip()

        # Next we need to group colors. This is more difficult than interiors as exterior colors can have very specific names. So we will keep quite a bit more. If it's less than 2% of the total it gets bucketed into other
        # Fill NaN values
        df['VehColorExt'] = df['VehColorExt'].fillna('not specified')
        frequency = df['VehColorExt'].value_counts(normalize= True)
        for color in df['VehColorExt'].values:
            if frequency[color] < 0.02:
                df['VehColorExt'] = df['VehColorExt'].replace(color, 'other')
    

    def clean_SellerRating():
        # Round seller ratings to get specific bins.
        df['SellerRating'] = df['SellerRating'].round()

        # Median for seller review count is 126 and first quartile is 28. Going to say if there's less than 25 reviews to throw it out.
        df['SellerRating'] = np.where(df['SellerRevCnt'] < 25, np.nan, df['SellerRating'])

        # Drop SellerRevCnt as it is not longer useful
        df.drop(columns = ['SellerRevCnt'])

    def clean_SellerListSrc():
        df['SellerListSrc'] = df['SellerListSrc'].fillna('not specified')
        frequency = df['SellerListSrc'].value_counts(normalize= True)
        for source in df['SellerListSrc'].values:
            if frequency[source] < 0.01:
                df['SellerListSrc'] = df['SellerListSrc'].replace(source, 'other')

    
    clean_VehHistory()
    clean_VehColorInt()
    clean_VehEngine()
    clean_VehDriveTrain()
    clean_VehColorExt()
    clean_SellerRating()
    clean_SellerListSrc()
    clean_SellerCity()



    # Define data transformation functions
    def add_log_data():
        numbers = df[['SellerRating', 'VehListdays', 'VehMileage', 'Dealer_Listing_Price', 'NumCylinders', 'CylinderSize']]
        for column in numbers.columns:
            if column == 'Dealer_Listing_Price':
                next
            else:
                numbers['log_of_' + column] = np.log(numbers[column])
                numbers['sqrt_of_' + column] = np.sqrt(numbers[column])
        
        numbers.replace([np.inf, -np.inf], np.nan, inplace = True)
        numbers - numbers.fillna(np.nan)

        return numbers


    def get_categories():
        categories = df[['SellerCity', 'SellerIsPriv', 'SellerListSrc', 'VehCertified', 'VehColorExt', 'VehFuel', 'VehYear', 'VehMake', 'VehModel', 'VehPriceLabel', 'Vehicle_Trim', 'IntColor1']]
        categories = pd.get_dummies(categories)

        binary_categories = df[['0 Owners', '1 Owner', '2 Owners', '3 Owners', '4 Owners', 'Accident(s) Reported', 'Buyback Protection Eligible', 'Non-Personal Use Reported', 'Title Issue(s) Reported', 'HEMI', 'AWD', '4WD', 'Tricoat', 'Metallic']]

        all_categories = pd.concat([categories, binary_categories], axis = 1)

        return all_categories
    
    # Get transformed dataframes
    log_data = add_log_data()
    category_data = get_categories()

    # Combine transformed dataframes
    features = pd.concat([log_data, category_data], axis = 1)


    def generalize_collinear_feats(df, threshold):
        '''
        Takes in a data frame and compares correlation coefficients between all variables. Removes any that are above the threshold.
        This will help generalize the model
        '''

        # Remove the listing price since we don't want to remove those correlations
        listing_price_copy = df['Dealer_Listing_Price']
        df.drop(columns = ['Dealer_Listing_Price'], inplace = True)

        # Get the correlation matrix
        correlations = df.corr()
        number_columns = range(len(correlations.columns) - 1)

        # Create an empty set to populate with highly correlated columns
        columns_to_drop = set()

        for i in number_columns:
            for j in range(i):
                item = correlations.iloc[j:(j+1), (i+1):(i+2)]
                columns = item.columns
                rows = item.index
                corr_val = abs(item.values)

                if corr_val > threshold:
                    # Print the correlated features and the correlation value
                    # print(columns.values[0], "|", rows.values[0], "|", round(corr_val[0][0], 2))
                    columns_to_drop.add(columns.values[0])
        

        df.drop(columns = columns_to_drop, inplace = True)
        df['Dealer_Listing_Price'] = listing_price_copy

        return df



    generalized_features = generalize_collinear_feats(features, 0.6)


    # Uncomment this to look at cleaned and transformed dataframe in csv
    # df.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Cleaned_Dataset.csv', index = False)
    # features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Transformed_Dataset.csv', index = False)
    # generalized_features.to_csv('F:/Documents/Projects/DataScienceChallenge/data/Generalized_Dataset.csv', index = False)
    

    return generalized_features

# Uncomment when testing this file
# get_sales_dataframe()
