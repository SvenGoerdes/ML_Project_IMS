import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
sys.path.append('../helper_functions')


# Custom transformer to update 'Carrier Type'
class IncoCarrierType(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting needed
    def transform(self, X):
        X = X.copy()
        X.loc[X['Carrier Name'] == 'SPECIAL FUNDS SEC 25-A', 'Carrier Type'] = '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)'
        return X

# Custom transformer to handle 'WCIO Part Of Body Code' and 'WCIO Part Of Body Description'
class IncoWCIOBodyCode(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X.loc[X['WCIO Part Of Body Code'] == -9, 'WCIO Part Of Body Code'] = 90
        X.loc[X['WCIO Part Of Body Code'] == 90, 'WCIO Part Of Body Description'] = 'MULTIPLE BODY PARTS'
        return X


# Custom transformer to handle zero birth year values

class IncoZeroBirthYEAR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X['Birth Year'] == 0, 'Birth Year'] = np.nan
        return X
    
# Custom transformer to handle 'Age at Injury'
class IncoZeroAgeAtInjury(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X.loc[X['Age at Injury'] == 0, 'Age at Injury'] = np.nan
        return X
    
class IncoFilterAgeAtInjury(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        X = X[(X['Age at Injury'] >= 14) | (X['Age at Injury'].isna())]
        return X


# Custom transformer to update 'Number of Dependents'
class IncoDependents(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X.loc[(X['Number of Dependents'] > 0) & (X['Age at Injury'].between(1, 15)), 'Number of Dependents'] = 0
        return X



# Custom transformer for accident date flagging and age check
class IncoCorrectAge(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['missing_values_flag'] = X[['Accident Date', 'Age at Injury', 'Birth Year']].isnull().any(axis=1)
        X['Age_BirthYear_Check'] = (
            (X['Age at Injury'] + X['Birth Year'] == X['Accident Date'].dt.year) |
            (X['Age at Injury'] + X['Birth Year'] == X['Accident Date'].dt.year - 1)
        )
        X.loc[~X['Age_BirthYear_Check'] & ~X['missing_values_flag'], 'Age at Injury'] -= 1
        X = X.drop(columns=['missing_values_flag', 'Age_BirthYear_Check'])
        return X
    

class IncoSwapAccidentDate(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        # Define the relevant date columns
        date_columns = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date']

        # Create a DataFrame with only the relevant date columns
        date_df = X[date_columns]

        # Find the smallest date across the specified columns for each row
        X['Min Date'] = date_df.min(axis=1, skipna=True)

        # Swap 'Accident Date' with the minimum date where applicable
        needs_swap = X['Min Date'] != X['Accident Date']

        # Apply swaps
        X.loc[needs_swap, 'Accident Date'] = X.loc[needs_swap, 'Min Date']

        # Adjust the other columns: if the smallest date came from another column, replace that column with the original 'Accident Date'
        for col in date_columns[1:]:  # Exclude 'Accident Date'
            swap_condition = X[col] == X['Min Date']
            X.loc[needs_swap & swap_condition, col] = X.loc[needs_swap & swap_condition, 'Accident Date']

        # Drop the helper column
        X.drop(columns='Min Date', inplace=True)
        return X



# Custom transformer for COVID-19 Indicator update
class IncoCovidIndicator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        # Change COVID-19 Indicator to 'N' if the Accident Date is before March 1, 2020
        X.loc[(X['COVID-19 Indicator'] == 'Y') & (X['Accident Date'] < "2020-03-01"), 'COVID-19 Indicator'] = 'N'
        return X

# Custom transformer for Gender
class IncoGenderNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X.loc[X['Gender']=='X','Gender'] = np.nan
        return X