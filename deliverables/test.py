# Note that in this script it only includes the code where there were changes applied

'''
=================================================================================
0. IMPORTS
=================================================================================
'''
import pandas as pd
import numpy as np

from category_encoders import TargetEncoder

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # for missing values

# data partition
from sklearn.model_selection import train_test_split

# Import helper_function, changed this because it didn't work the same way as notebook
import sys
sys.path.append('helper_functions')
from helper_functions import *

# Get the current working directory
import os
current_directory = os.getcwd()
print("Current Directory:", current_directory)

'''
=================================================================================
=================================================================================
'''


'''
=================================================================================
1. LOAD DATASET
=================================================================================
'''

WCB_original = pd.read_csv('project_data/train_data.csv', delimiter=',',dtype={'Zip Code': str})
X_test = pd.read_csv('project_data/test_data.csv', delimiter=',',dtype={'Zip Code': str})

'''
=================================================================================
=================================================================================
'''


'''
=================================================================================
3.1 UNDERSTANDING THE DATASET STRUCTURE
=================================================================================
'''
# (1) Creating WCB from the original --------------------------------------------
WCB = WCB_original.copy()

# (2) Drop duplicate ------------------------------------------------------------
WCB = WCB.drop(index=257901)

# (3) Set Claim Indentifier as Index --------------------------------------------
WCB.set_index('Claim Identifier', inplace=True)
X_test.set_index('Claim Identifier', inplace=True)

# (4) Drop null values from Claim Injury Type -----------------------------------
WCB = WCB.dropna(subset=['Claim Injury Type'])

# (5) Drop column OIICS Nature of Injury Description ----------------------------
WCB = WCB.drop(columns=['OIICS Nature of Injury Description'])
X_test =  X_test.drop(columns=['OIICS Nature of Injury Description'])

# (6) Drop another duplicate ----------------------------------------------------
WCB = WCB.drop(index=5686771)

# (7) Check for duplicates, excluding columns at a time -------------------------
# List of columns to check for duplicates
columns_to_check = WCB.columns.tolist()

# (8) Iterate over each column, excluding one at a time
for col in columns_to_check:
    # Define the subset of columns to check in this iteration (excluding 'col')
    cols_to_check_now = [c for c in columns_to_check if c != col]

    # Identify duplicates based on these columns
    duplicates = WCB[WCB.duplicated(subset=cols_to_check_now, keep=False)]

    #Drop duplicates, keeping the first occurrence in each subset where one column can differ
    WCB = WCB.drop_duplicates(subset=cols_to_check_now, keep='first')

# (9) Drop column WCB Decision ------------------------------------------------------
WCB = WCB.drop(columns=['WCB Decision'])

# (10) Setting unknown values to nan -------------------------------------------------
unknown_values = {'Alternative Dispute Resolution': 'U',   'Carrier Type': 'UNKNOWN', 'County of Injury': 'UNKNOWN',
    'Gender': 'U','Medical Fee Region': 'UK'}

WCB.replace(unknown_values, np.nan, inplace=True)

'''
=================================================================================
=================================================================================
'''


'''
=================================================================================
3.3 CHECKING INCOHERENCIES
=================================================================================
'''
# 3.3.1 Carrier Name & Carrier Type ---------------------------------------------
# Update the Carrier Type for rows where Carrier Name is 'SPECIAL FUNDS SEC 25-A'
WCB.loc[WCB['Carrier Name'] == 'SPECIAL FUNDS SEC 25-A', 'Carrier Type'] = '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)'

#3.3.5 WCIO Part of Body Code & Description -------------------------------------
# Update the rows where 'WCIO Part Of Body Code' is -9
WCB.loc[WCB['WCIO Part Of Body Code'] == -9, 'WCIO Part Of Body Code'] = 90
WCB.loc[WCB['WCIO Part Of Body Code'] == 90, 'WCIO Part Of Body Description'] = 'MULTIPLE BODY PARTS'

# Do the same for the test data
X_test.loc[X_test['WCIO Part Of Body Code'] == -9, 'WCIO Part Of Body Code'] = 90
X_test.loc[X_test['WCIO Part Of Body Code'] == 90, 'WCIO Part Of Body Description'] = 'MULTIPLE BODY PARTS'

# 3.3.6 Average Weekly Wage -----------------------------------------------------
# Replace all instances of Average Weekly Wage == 0 with NaN
WCB.loc[WCB['Average Weekly Wage'] == 0, 'Average Weekly Wage'] = np.nan

# 3.3.7 Birth Year --------------------------------------------------------------
# Replace all instances of Birth Year == 0 with NaN
WCB.loc[WCB['Birth Year'] == 0, 'Birth Year'] = np.nan
X_test.loc[X_test['Birth Year'] == 0, 'Birth Year'] = np.nan

# 3.3.8 Number of Dependents -----------------------------------------------------
# Changing number of dependents of people with age under 16 to zero
WCB.loc[(WCB['Number of Dependents'] > 0) & (WCB['Age at Injury'].between(1, 15)),'Number of Dependents']= 0

# 3.3.11 Assembly Date ------------------------------------------------------------
# Swap Accident Date and Assembly Date for invalid entries
invalid_entries_acc_ass=invalid_entries(WCB,'Accident Date','Assembly Date')
WCB.loc[invalid_entries_acc_ass.index, ['Accident Date', 'Assembly Date']] = WCB.loc[invalid_entries_acc_ass.index, ['Assembly Date', 'Accident Date']].values

'''
=================================================================================
=================================================================================
'''


'''
=================================================================================
5. DATA SPLITTING
=================================================================================
'''
X = WCB.drop(columns=['Claim Injury Type'])
y = WCB[['Claim Injury Type']]
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                 test_size = 0.3,
                                                 shuffle = True,
                                                 random_state = 0,
                                                 stratify = y)
'''
=================================================================================
=================================================================================
'''


'''
=================================================================================
5.1 HANDLING MISSING VALUES
=================================================================================
'''

# 5.1.1 IME-4 Count--------------------------------------------------------------
X_train['IME-4 Count'] = X_train['IME-4 Count'].fillna(0)
X_val['IME-4 Count'] = X_val['IME-4 Count'].fillna(0)
X_test['IME-4 Count'] = X_test['IME-4 Count'].fillna(0)

# 5.1.3 C-2, C-3 Date -----------------------------------------------------------
X_train['Date Difference'] = X_train.apply(
    lambda row: row['C-2 Date'] - row['Accident Date'] if row['C-2 Date'] > row['Accident Date'] else pd.NaT, axis=1
)

# Calculate the median of the differences in days for X_train
median_difference_2 = X_train['Date Difference'].dropna().median().days
X_train = X_train.drop(columns=['Date Difference'])

X_train = impute_dates_with_difference(X_train, 'C-2 Date', 'Accident Date', 'Accident Date', median_difference_2)
X_val = impute_dates_with_difference(X_val, 'C-2 Date', 'Accident Date', 'Accident Date', median_difference_2)
X_test = impute_dates_with_difference(X_test, 'C-2 Date', 'Accident Date', 'Accident Date', median_difference_2)


X_train['Date Difference'] = X_train.apply(
    lambda row: row['C-3 Date'] - row['Accident Date'] if row['C-3 Date'] > row['Accident Date'] else pd.NaT, axis=1
)

# Calculate the median of the differences in days for X_train
median_difference_3 = X_train['Date Difference'].dropna().median().days
X_train = X_train.drop(columns=['Date Difference'])

X_train = impute_dates_with_difference(X_train, 'C-3 Date', 'Accident Date', 'Accident Date', median_difference_3)
X_val = impute_dates_with_difference(X_val, 'C-3 Date', 'Accident Date', 'Accident Date', median_difference_3)
X_test = impute_dates_with_difference(X_test, 'C-3 Date', 'Accident Date', 'Accident Date', median_difference_3)

# -------------------------------------------------------------------------------

