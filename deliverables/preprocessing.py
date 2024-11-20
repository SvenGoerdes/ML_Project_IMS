# Note that in this script it only includes the code where there were changes applied


# ================================================================================
# 0. IMPORTS
# ================================================================================


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
sys.path.append('../helper_functions')
from helper_functions import *

# Get the current working directory
import os
# redundant this is dependent on where you execute the script from 
# current_directory = os.getcwd()
# print("Current Directory:", current_directory)



# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================




# ================================================================================
# 1. LOAD DATASET
# ================================================================================


WCB_original = pd.read_csv('project_data/train_data.csv', delimiter=',',dtype={'Zip Code': str})
X_test = pd.read_csv('project_data/test_data.csv', delimiter=',',dtype={'Zip Code': str})


# ================================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================



# ================================================================================
# 3.1 UNDERSTANDING THE DATASET STRUCTURE
# ================================================================================

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

# (6) Convert to datetime -------------------------------------------------------
date_columns = ['Accident Date', 'Assembly Date','C-2 Date', 'C-3 Date', 'First Hearing Date']
# Convert columns to datetime
for column in date_columns:
    WCB[column] = pd.to_datetime(WCB[column], format='%Y-%m-%d', errors='coerce')
    X_test[column] = pd.to_datetime(X_test[column], format='%Y-%m-%d', errors='coerce')



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


# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================




# =================================================================================
# 3.3 CHECKING INCOHERENCIES
# =================================================================================

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
invalid_entries_acc_ass=invalid_entries(WCB,'Accident Date','Assembly Date') # type: ignore
WCB.loc[invalid_entries_acc_ass.index, ['Accident Date', 'Assembly Date']] = WCB.loc[invalid_entries_acc_ass.index, ['Assembly Date', 'Accident Date']].values


# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================




# ================================================================================
# 5. DATA SPLITTING
# ================================================================================

X = WCB.drop(columns=['Claim Injury Type'])
y = WCB[['Claim Injury Type']]
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                 test_size = 0.3,
                                                 shuffle = True,
                                                 random_state = 0,
                                                 stratify = y)

# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================




# ================================================================================
# 5.1 HANDLING MISSING VALUES
# ================================================================================

# 5.1.1 IME-4 Count--------------------------------------------------------------
X_train['IME-4 Count'] = X_train['IME-4 Count'].fillna(0)
X_val['IME-4 Count'] = X_val['IME-4 Count'].fillna(0)
X_test['IME-4 Count'] = X_test['IME-4 Count'].fillna(0)

# 5.1.3 C-2 Date ----------------------------------------------------------------

X_train['Date Difference'] = X_train.apply(
    lambda row: row['C-2 Date'] - row['Accident Date'] if row['C-2 Date'] > row['Accident Date'] else pd.NaT, axis=1
)

# Calculate the median of the differences in days for X_train
median_difference_2 = X_train['Date Difference'].dropna().median().days
X_train = X_train.drop(columns=['Date Difference'])

X_train = impute_dates_with_difference(X_train, 'C-2 Date', 'Accident Date', 'Accident Date', median_difference_2)
X_val = impute_dates_with_difference(X_val, 'C-2 Date', 'Accident Date', 'Accident Date', median_difference_2)
X_test = impute_dates_with_difference(X_test, 'C-2 Date', 'Accident Date', 'Accident Date', median_difference_2)

# 5.1.3 C-3 Date -----------------------------------------------------------------

X_train['Date Difference'] = X_train.apply(
    lambda row: row['C-3 Date'] - row['Accident Date'] if row['C-3 Date'] > row['Accident Date'] else pd.NaT, axis=1
)

# Calculate the median of the differences in days for X_train
median_difference_3 = X_train['Date Difference'].dropna().median().days
X_train = X_train.drop(columns=['Date Difference'])

X_train = impute_dates_with_difference(X_train, 'C-3 Date', 'Accident Date', 'Accident Date', median_difference_3)
X_val = impute_dates_with_difference(X_val, 'C-3 Date', 'Accident Date', 'Accident Date', median_difference_3)
X_test = impute_dates_with_difference(X_test, 'C-3 Date', 'Accident Date', 'Accident Date', median_difference_3)


# 5.1.4 Birth Year -----------------------------------------------------------------
# Rows to replace in X_train
filtered_rows_train =X_train[(X_train['Birth Year'].isna()) & (X_train['Accident Date'].notna()) & (X_train['Age at Injury'].notna())]

X_train.loc[filtered_rows_train.index, 'Birth Year'] = (
    X_train.loc[filtered_rows_train.index, 'Accident Date'].dt.year - X_train.loc[filtered_rows_train.index, 'Age at Injury']
)

# Rows to replace in X_val
filtered_rows_val= X_val[(X_val['Birth Year'].isna()) & (X_val['Accident Date'].notna()) & (X_val['Age at Injury'].notna())]

X_val.loc[filtered_rows_val.index, 'Birth Year'] = (
    X_val.loc[filtered_rows_val.index, 'Accident Date'].dt.year - X_val.loc[filtered_rows_val.index, 'Age at Injury']
)

# Rows to replace in X_test
filtered_rows_test= X_test[(X_test['Birth Year'].isna()) & (X_test['Accident Date'].notna()) & (X_test['Age at Injury'].notna())]

X_test.loc[filtered_rows_test.index, 'Birth Year'] = (
    X_test.loc[filtered_rows_test.index, 'Accident Date'].dt.year - X_test.loc[filtered_rows_test.index, 'Age at Injury']
)

# Calculate the median of Birth Year
birth_median_train= X_train['Birth Year'].median()
#Impute missing values
X_train.loc[X_train['Birth Year'].isna(),'Birth Year']= birth_median_train
X_val.loc[X_val['Birth Year'].isna(), 'Birth Year'] = birth_median_train
X_test.loc[X_test['Birth Year'].isna(), 'Birth Year'] = birth_median_train

# 5.1.5 Medical Fee Region ------------------------------------------------------
# Extract proportions as a dictionary to ensure correct association
prop = X_train['Medical Fee Region'].value_counts(normalize=True)
regions = prop.index.tolist()  
proportions_train = prop.values 

print("Proportions array:", proportions_train)
print("Regions list:", regions)

# Impute missing values in X_train and X_val
X_train = impute_prop(regions, proportions_train, X_train, 'Medical Fee Region')
X_val = impute_prop(regions, proportions_train, X_val, 'Medical Fee Region')

# Impute missing values in X_test | redundant as we are dropping the region column
X_test = impute_prop(regions, proportions_train, X_test, 'Medical Fee Region')

# 5.1.6 Zip Code ----------------------------------------------------------------
X_train.fillna({'Zip Code':'UNKNOWN'}, inplace=True)
X_val.fillna({'Zip Code':'UNKNOWN'}, inplace=True)

# Redundant as we are dropping the zip column
X_test.fillna({'Zip Code':'UNKNOWN'}, inplace=True)

# 5.1.7 WCIO Part Of Body Code & Description -----------------------------------
default_value_code = 100
default_value_desc = "Unknown"

# Columns with codes and descriptions
columns_code = ['WCIO Part Of Body Code', 'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code']
columns_desc = ['WCIO Part Of Body Description', 'WCIO Cause of Injury Description', 'WCIO Nature of Injury Description']

# check if all thse columns are NaN for X_train and X_val
missing_condition_train = X_train[columns_code + columns_desc].isna().all(axis=1)
missing_condition_val = X_val[columns_code + columns_desc].isna().all(axis=1)
missing_condition_test = X_test[columns_code + columns_desc].isna().all(axis=1)

# Apply default values to X_train and X_val where conditions are met
for col in columns_code:
    X_train.loc[missing_condition_train, col] = default_value_code
    X_val.loc[missing_condition_val, col] = default_value_code
    X_test.loc[missing_condition_test, col] = default_value_code

for col in columns_desc:
    X_train.loc[missing_condition_train, col] = default_value_desc
    X_val.loc[missing_condition_val, col] = default_value_desc
    X_test.loc[missing_condition_test, col] = default_value_desc

X_train = impute_with(X_train, 'WCIO Part Of Body Code', 'WCIO Cause of Injury Code', metric='mode')
X_val = impute_with(X_val, 'WCIO Part Of Body Code', 'WCIO Cause of Injury Code',reference_df=X_train, metric='mode')
X_test = impute_with(X_test, 'WCIO Part Of Body Code', 'WCIO Cause of Injury Code',reference_df=X_train, metric='mode')

# DataFrame with unique WCIO Part Of Body Codes and their descriptions (without missing descriptions)
body_code_description_train = X_train[['WCIO Part Of Body Code', 'WCIO Part Of Body Description']].drop_duplicates()
body_code_description_train = body_code_description_train[body_code_description_train['WCIO Part Of Body Description'].notna()]

# Dictionary to map WCIO Part Of Body Code to its description
description_dict = dict(zip(body_code_description_train['WCIO Part Of Body Code'], body_code_description_train['WCIO Part Of Body Description']))

# Fill missing descriptions in X_train, X_val, X_test using the mapping | This is redundant as we dropo the description column later
X_train['WCIO Part Of Body Description'] = X_train['WCIO Part Of Body Description'].fillna(X_train['WCIO Part Of Body Code'].map(description_dict))
X_val['WCIO Part Of Body Description'] = X_val['WCIO Part Of Body Description'].fillna(X_val['WCIO Part Of Body Code'].map(description_dict))
X_test['WCIO Part Of Body Description'] = X_test['WCIO Part Of Body Description'].fillna(X_test['WCIO Part Of Body Code'].map(description_dict))

# 5.1.8 WCIO Nature of Injury Code & Description --------------------------------
X_train = impute_with(X_train, 'WCIO Nature of Injury Code', 'WCIO Part Of Body Code', metric='mode')
X_val = impute_with(X_val, 'WCIO Nature of Injury Code', 'WCIO Part Of Body Code',reference_df=X_train, metric='mode')
X_test = impute_with(X_test, 'WCIO Nature of Injury Code', 'WCIO Part Of Body Code',reference_df=X_train, metric='mode')

# DataFrame with unique WCIO Nature of Injury Codes and their descriptions (in X_train and without missing descriptions)
nature_code_description_train = X_train[['WCIO Nature of Injury Code', 'WCIO Nature of Injury Description']].drop_duplicates()
nature_code_description_train = nature_code_description_train[nature_code_description_train['WCIO Nature of Injury Description'].notna()]

# Dictionary to map WCIO Nature of Injury Code to its description
description_dict = dict(zip(nature_code_description_train['WCIO Nature of Injury Code'], nature_code_description_train['WCIO Nature of Injury Description']))

# Fill missing descriptions in the original DataFrame using the mapping | redundant as we drop the values later 
X_train['WCIO Nature of Injury Description'] = X_train['WCIO Nature of Injury Description'].fillna(X_train['WCIO Nature of Injury Code'].map(description_dict))
X_val['WCIO Nature of Injury Description'] = X_val['WCIO Nature of Injury Description'].fillna(X_val['WCIO Nature of Injury Code'].map(description_dict))
X_test['WCIO Nature of Injury Description'] = X_test['WCIO Nature of Injury Description'].fillna(X_test['WCIO Nature of Injury Code'].map(description_dict))

# 5.1.9 WCIO Cause of Injury Code & Description ---------------------------------
X_train = impute_with(X_train, 'WCIO Cause of Injury Code', 'WCIO Part Of Body Code', metric='mode')
X_val = impute_with(X_val, 'WCIO Cause of Injury Code', 'WCIO Part Of Body Code',reference_df=X_train, metric='mode')
X_test = impute_with(X_test, 'WCIO Cause of Injury Code', 'WCIO Part Of Body Code',reference_df=X_train, metric='mode')

# DataFrame with unique WCIO Cause of Injury Codes and their descriptions (in X_train and without missing descriptions)
cause_code_description_train = X_train[['WCIO Cause of Injury Code', 'WCIO Cause of Injury Description']].drop_duplicates()
cause_code_description_train = cause_code_description_train[cause_code_description_train['WCIO Cause of Injury Description'].notna()]

# Dictionary to map WCIO Cause of Injury Code to its description (based on X_train)
description_dict_train = dict(zip(cause_code_description_train['WCIO Cause of Injury Code'], cause_code_description_train['WCIO Cause of Injury Description']))

# Fill missing descriptions in X_train and X_val using the mapping
X_train['WCIO Cause of Injury Description'] = X_train['WCIO Cause of Injury Description'].fillna(X_train['WCIO Cause of Injury Code'].map(description_dict_train))
X_val['WCIO Cause of Injury Description'] = X_val['WCIO Cause of Injury Description'].fillna(X_val['WCIO Cause of Injury Code'].map(description_dict_train))
X_test['WCIO Cause of Injury Description'] = X_test['WCIO Cause of Injury Description'].fillna(X_test['WCIO Cause of Injury Code'].map(description_dict_train))

# 5.1.10 Industry Code & Description --------------------------------------------
# Extract proportions for the top 10 Industry Code values only
top_industries = X_train['Industry Code'].value_counts(normalize=True).head(10)
industries = top_industries.index.tolist() 
proportions_train = top_industries.values   
proportions_train = proportions_train / proportions_train.sum()

# Print debug information
print("Top 10 Proportions array:", proportions_train)
print("Top 10 Industry Codes list:", industries)

# Impute missing values in X_train and X_val
X_train = impute_prop(industries, proportions_train, X_train, 'Industry Code')
X_val = impute_prop(industries, proportions_train, X_val, 'Industry Code')
X_test = impute_prop(industries, proportions_train, X_test, 'Industry Code')

# DataFrame with unique Industry Codes and their descriptions (in X_train and without missing descriptions)
industry_code_description_train = X_train[['Industry Code', 'Industry Code Description']].drop_duplicates()
industry_code_description_train = industry_code_description_train[industry_code_description_train['Industry Code Description'].notna()]

# Dictionary to map Industry Code to its description
description_dict_train = dict(zip(industry_code_description_train['Industry Code'], industry_code_description_train['Industry Code Description']))

# Fill missing descriptions in X_train using the mapping
X_train['Industry Code Description'] = X_train['Industry Code Description'].fillna(X_train['Industry Code'].map(description_dict_train))
X_val['Industry Code Description'] = X_val['Industry Code Description'].fillna(X_val['Industry Code'].map(description_dict_train))
X_test['Industry Code Description'] = X_test['Industry Code Description'].fillna(X_test['Industry Code'].map(description_dict_train))

# 5.1.11 Average Weekly Wage
# fill zero with na 
X_test['Average Weekly Wage'] = X_test['Average Weekly Wage'].replace(0, np.nan)

# impute for training dataset
X_train['Average Weekly Wage'] = X_train.groupby('Industry Code', observed=False)['Average Weekly Wage'].transform(lambda x: x.fillna(x.mean()))
# calculate mean for each industry code in training dataset
industry_mean_train = X_train.groupby('Industry Code')['Average Weekly Wage'].mean()

# impute for validation dataset
X_val['Average Weekly Wage'] = X_val.groupby('Industry Code')['Average Weekly Wage'].transform(
    lambda x: x.fillna(industry_mean_train.get(x.name, float('nan')))
)

# impute for test dataset
X_test['Average Weekly Wage'] = X_test.groupby('Industry Code')['Average Weekly Wage'].transform(
    lambda x: x.fillna(industry_mean_train.get(x.name, float('nan')))
)

# 5.1.12 Accident Date ----------------------------------------------------------
# Difference between Assembly Date and Accident Date for X_train (only for valid cases)
X_train['Date Difference'] = X_train.apply(
    lambda row: row['Assembly Date'] - row['Accident Date'] if row['Assembly Date'] > row['Accident Date'] else pd.NaT, axis=1
)

# Calculate the median of the differences in days for X_train
median_difference_train = X_train['Date Difference'].dropna().median().days

# Condition to impute missing Accident Dates in X_train and X_val
condition_train = (X_train['Accident Date'].isna()) & (X_train['Assembly Date'].notna())
condition_val = (X_val['Accident Date'].isna()) & (X_val['Assembly Date'].notna())
condition_test = (X_test['Accident Date'].isna()) & (X_test['Assembly Date'].notna())

# Impute Accident Date for X_train
X_train.loc[condition_train, 'Accident Date'] = X_train.loc[condition_train, 'Assembly Date'] - pd.Timedelta(days=median_difference_train)
X_val.loc[condition_val, 'Accident Date'] = X_val.loc[condition_val, 'Assembly Date'] - pd.Timedelta(days=median_difference_train)
X_test.loc[condition_test, 'Accident Date'] = X_test.loc[condition_test, 'Assembly Date'] - pd.Timedelta(days=median_difference_train)

X_train = X_train.drop(columns=['Date Difference'])

# 5.1.13 Age at Injury
# Rows to replace
filtered_rows_train = X_train[(X_train['Age at Injury'].isna()) & (X_train['Accident Date'].notna())]

# Calculate and replace Age at Injury in X_train
X_train.loc[filtered_rows_train.index, 'Age at Injury'] = (
    X_train.loc[filtered_rows_train.index, 'Accident Date'].dt.year - X_train.loc[filtered_rows_train.index, 'Birth Year']
)

#Repeat the process for X_val
filtered_rows_val = X_val[(X_val['Age at Injury'].isna()) & (X_val['Accident Date'].notna())]

X_val.loc[filtered_rows_val.index, 'Age at Injury'] = (
    X_val.loc[filtered_rows_val.index, 'Accident Date'].dt.year - X_val.loc[filtered_rows_val.index, 'Birth Year']
)

#Repeat the process for X_test
filtered_rows_test = X_test[(X_test['Age at Injury'].isna()) & (X_test['Accident Date'].notna())]

X_test.loc[filtered_rows_test.index, 'Age at Injury'] = (
    X_test.loc[filtered_rows_test.index, 'Accident Date'].dt.year - X_test.loc[filtered_rows_test.index, 'Birth Year']
)


# Calculate the difference in days for X_train
X_train['Date Difference'] = (X_train['Assembly Date'] - X_train['Accident Date']).dt.days

# Compute the median and mean of the difference in days for X_train
median_diff_train = X_train['Date Difference'].median()
mean_diff_train = X_train['Date Difference'].mean()

X_train = X_train.drop(columns=['Date Difference'])
median_diff_train, mean_diff_train,


# Rows to replace in X_train
filtered_rows_train = X_train[(X_train['Age at Injury'].isna()) & (X_train['Accident Date'].isna()) & (X_train['Assembly Date'].notna())]

# Calculate and replace Age at Injury in X_train
X_train.loc[filtered_rows_train.index, 'Age at Injury'] = (
    X_train.loc[filtered_rows_train.index, 'Assembly Date'].dt.year - X_train.loc[filtered_rows_train.index, 'Birth Year']
)

# Doing the same for X_val
filtered_rows_val = X_val[(X_val['Age at Injury'].isna()) & (X_val['Accident Date'].isna()) & (X_val['Assembly Date'].notna())]

X_val.loc[filtered_rows_val.index, 'Age at Injury'] = (
    X_val.loc[filtered_rows_val.index, 'Assembly Date'].dt.year - X_val.loc[filtered_rows_val.index, 'Birth Year']
)

# Doing the same for X_test 
filtered_rows_test = X_test[(X_test['Age at Injury'].isna()) & (X_test['Accident Date'].isna()) & (X_test['Assembly Date'].notna())]

X_test.loc[filtered_rows_test.index, 'Age at Injury'] = (
    X_test.loc[filtered_rows_test.index, 'Assembly Date'].dt.year - X_test.loc[filtered_rows_test.index, 'Birth Year']
)

# 5.1.14 Gender -----------------------------------------------------------------
# impute with mode based on the Industry Code
X_train = impute_with(X_train, 'Gender', 'Industry Code', ['X'], metric = 'mode')

# Use reference DataFrame X_train to impute missing values in X_val to avoid data leakage
X_val = impute_with(X_val, 'Gender', 'Industry Code', ['X'], reference_df=X_train, metric = 'mode')

# Use reference DataFrame X_train to impute missing values in X_test to avoid data leakage
X_test = impute_with(X_test, 'Gender', 'Industry Code', ['U','X'], reference_df = X_train, metric = 'mode')

# 5.1.15 Carrier Type -----------------------------------------------------------
X_train = impute_with(X_train, target_column='Carrier Type',group_column='Industry Code', metric='mode')
X_val = impute_with(X_val, target_column='Carrier Type', group_column='Industry Code',reference_df=X_train, metric='mode')
X_test = impute_with(X_test, target_column='Carrier Type', group_column='Industry Code',reference_df=X_train, metric='mode')

# 5.1.16 County of Injury
X_train = impute_with(df=X_train, target_column='County of Injury', group_column='District Name', metric='mode')
X_val= impute_with(df=X_val, target_column='County of Injury', group_column='District Name',reference_df=X_train, metric='mode')
X_test= impute_with(df=X_test, target_column='County of Injury', group_column='District Name',reference_df=X_train, metric='mode')

# 5.1.17 Alternative Dispute Resolution
# Extract proportions for ADR values
top_industries = X_train['Alternative Dispute Resolution'].value_counts(normalize=True)
adr = top_industries.index.tolist()
proportions_train = top_industries.values  

print("Proportions array:", proportions_train)
print("Industry Codes list:", adr)

# Impute missing values in X_train and X_val
X_train = impute_prop(adr, proportions_train, X_train, 'Alternative Dispute Resolution')
X_val = impute_prop(adr, proportions_train, X_val, 'Alternative Dispute Resolution')
# this is redundant with our current dataset
X_test = impute_prop(adr, proportions_train, X_test, 'Alternative Dispute Resolution')

#Drop Agreement Reached
X_train = X_train.drop(columns=['Agreement Reached'])
X_val = X_val.drop(columns=['Agreement Reached'])

# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================


# ================================================================================
# 5.2 Outlier Detection and Treatment
# ================================================================================
# We decided to create a copy of X_train and X_val to avoid mistakes.
X_train_copy = X_train.copy() 
X_val_copy = X_val.copy()

# 5.2.1 Date Outliers -----------------------------------------------------------
# Create a list with all the date variables
dates=['Accident Date','Assembly Date','C-2 Date','C-3 Date','First Hearing Date']

# List with the date columns with outliers
dates_with_outliers = ['Accident Date','C-2 Date','C-3 Date']

for date in dates_with_outliers:
  X_train_copy = X_train_copy[X_train_copy[date].ge('2018-01-01') | X_train_copy[date].isna()]
  X_val_copy = X_val_copy[X_val_copy[date].ge('2018-01-01') | X_val_copy[date].isna()]


# Creating a column in X_train_copy and X_val_copy that calculates the days between Accident Date and Assembly Date
X_train_copy.loc[:, 'days_between_accident_assembly'] = days_between(X_train_copy, 'Accident Date', 'Assembly Date')
X_val_copy.loc[:, 'days_between_accident_assembly'] = days_between(X_val_copy, 'Accident Date', 'Assembly Date')

# Add log transformation column in X_train
X_train_copy['log_days_between_accident_assembly'] = np.log1p(X_train_copy['days_between_accident_assembly'])

# Using the remove_outliers_iqr function created in helper_functions we'll 
lower, upper, X_train_copy = log_remove_outliers_iqr(X_train_copy,'log_days_between_accident_assembly',2)

# Applying the same upper bound to X_val_copy
initial_count = X_val_copy.shape[0]

X_val_copy = X_val_copy[X_val_copy['days_between_accident_assembly']<upper]

final_count = X_val_copy.shape[0]

# Dropping the columns created
X_train_copy = X_train_copy.drop('days_between_accident_assembly', axis=1)
X_train_copy = X_train_copy.drop('log_days_between_accident_assembly', axis=1)
X_val_copy = X_val_copy.drop('days_between_accident_assembly', axis=1)

# 5.2.2 Numerical outliers ------------------------------------------------------
# Selecting only the numerical columns
numerical_cols = ['Age at Injury','Average Weekly Wage','Birth Year','IME-4 Count','Number of Dependents']

# Remove outlier in X_train, obtain lower and upper bound and apply to X_val
lower, upper, X_train_copy = remove_outliers_iqr(X_train_copy,'Age at Injury')

# Apply the same upper bound to X_val_copy
X_val_copy = X_val_copy[X_val_copy['Age at Injury']<upper]

# Apply minimum of 11 years old to X_train_copy and X_val_copy
X_train_copy = X_train_copy[X_train_copy['Age at Injury']>11]
X_val_copy = X_val_copy[X_val_copy['Age at Injury']>=11]

# Average Weekly Wage -----------------------------------------------------------
percentile_99 = X_train_copy['Average Weekly Wage'].quantile(0.9999)

# Apply conditions to the training set first
X_train_copy = X_train_copy[(X_train_copy['Average Weekly Wage'] < percentile_99)]

# Apply the same conditions to the validation set
X_val_copy = X_val_copy[(X_val_copy['Average Weekly Wage'] < percentile_99)]

# Add log transformation column
X_train_copy['log_avg_weekly_wage'] = np.log1p(X_train_copy['Average Weekly Wage'])

# Threshold of 2 for a more conservative range
lower,upper,X_train_copy = log_remove_outliers_iqr(X_train_copy,'log_avg_weekly_wage',2)

# Apply upper bound to X_train_copy and X_val_copy
X_val_copy = X_val_copy[(X_val_copy['Average Weekly Wage'] < upper)]

# IME-4 Count -------------------------------------------------------------------
X_train_copy = X_train_copy[(X_train_copy['IME-4 Count'] < 40)]
X_val_copy = X_val_copy[(X_val_copy['IME-4 Count'] < 40)]

# 5.2.3 Applying Changes --------------------------------------------------------
X_train = X_train_copy.copy()
X_val = X_val_copy.copy()

# after we have cleaned the data we need to exclude the index out of the y_train and y_val 
y_train = y_train.loc[X_train.index]
y_val = y_val.loc[X_val.index]




# ================================================================================ save the data 
# X_train.to_csv('project_data/X_train.csv')
# X_val.to_csv('project_data/X_val.csv')
# X_test.to_csv('project_data/X_test.csv')
# y_train.to_csv('project_data/y_train.csv')

