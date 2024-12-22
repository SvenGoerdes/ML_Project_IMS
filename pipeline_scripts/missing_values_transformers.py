
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import numpy as np



from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting required, just return self
        return self

    def transform(self, X):
        """
        Convert numpy array X back to a DataFrame without the need for column names.
        """
        return pd.DataFrame(X)

class ImputeBirthYearFromAccident(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # This transformer doesn't require fitting, just transforming
        return self

    def transform(self, X):
        X = X.copy()
        
        # Step 1: Impute 'Birth Year' using 'Accident Date' and 'Age at Injury' if 'Birth Year' is missing
        filtered_rows = X[(X['Birth Year'].isna()) & (X['Accident Date'].notna()) & (X['Age at Injury'].notna())]

        X.loc[filtered_rows.index, 'Birth Year'] = (
            X.loc[filtered_rows.index, 'Accident Date'].dt.year - X.loc[filtered_rows.index, 'Age at Injury']
        )

        return X


class ImputeBirthYearWithMedian(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Step 2: Calculate the median 'Age at Injury' from non-missing rows
        self.median_age_at_injury = X['Age at Injury'].median()
        
        # Calculate the median 'Birth Year' from existing non-missing values
        self.birth_median_train = X['Birth Year'].median()  
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Step 3: Impute 'Birth Year' using 'Accident Date' and 'Age at Injury' where possible
        X.loc[X['Birth Year'].isna(), 'Birth Year'] = (
            X.loc[X['Birth Year'].isna(), 'Accident Date'].dt.year - self.median_age_at_injury
        )
        
        # Step 4: If 'Birth Year' is still missing, fill it with the median
        X.loc[X['Birth Year'].isna(), 'Birth Year'] = self.birth_median_train
        
        return X



class ImputeProportionalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        """
        Custom transformer for proportional imputation of missing values.
        
        Args:
            column (str): The column to impute.
        """
        self.column = column

    def fit(self, X, y=None):
        # Calculate the proportions of the values in the column
        prop = X[self.column].value_counts(normalize=True)
        self.categories = prop.index.tolist()
        self.proportions = prop.values
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Find the indices of missing values in the specified column
        missing_indices = X_copy[self.column].isna()
        
        # Generate imputed values based on the calculated proportions from before
        imputed_values = np.random.choice(self.categories, size=missing_indices.sum(), p=self.proportions)
        
        # Fill in the missing values with the imputed values
        X_copy.loc[missing_indices, self.column] = imputed_values
        
        return X_copy
    

class ImputeProportionalTransformerColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        """
        Custom transformer for proportional imputation of missing values.
        
        Args:
            column (str): The column to impute.
        """
        self.column = column

    def fit(self, X, y=None):
        """
        Calculates the proportions of the values in the specified column.
        """
        # Calculate the proportions of the values in the column
        prop = X[self.column].value_counts(normalize=True)
        self.categories = prop.index.tolist()
        self.proportions = prop.values
        return self

    def transform(self, X):
        """
        Creates a new column with imputed values for the missing entries.
        """
        X_copy = X.copy()
        
        # Find the indices of missing values in the specified column
        missing_indices = X_copy[self.column].isna()
        
        # Generate imputed values based on the calculated proportions from before
        imputed_values = np.random.choice(self.categories, size=missing_indices.sum(), p=self.proportions)
        
        # Create a new column with the name '<column> + imputed' and copy the original values
        imputed_column_name = f"{self.column}_Imputed"
        X_copy[imputed_column_name] = X_copy[self.column]
        
        # Fill in the missing values in the new column with the imputed values
        X_copy.loc[missing_indices, imputed_column_name] = imputed_values
        
        return X_copy



class FillMissingAllWCIOWithUnknown(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary for this transformation
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Define the columns and default values directly here
        columns_code = ['WCIO Part Of Body Code', 'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code']
        columns_desc = ['WCIO Part Of Body Description', 'WCIO Cause of Injury Description', 'WCIO Nature of Injury Description']
        
        default_value_code = 100
        default_value_desc = "Unknown"

        # Identify rows where all specified code and description columns are missing
        missing_condition = X_copy[columns_code + columns_desc].isna().all(axis=1)
        
        # Fill those rows with default values
        X_copy.loc[missing_condition, columns_code] = default_value_code
        X_copy.loc[missing_condition, columns_desc] = default_value_desc
        
        return X_copy
    
from sklearn.base import BaseEstimator, TransformerMixin

class FillMissingDescriptionsWithCode(BaseEstimator, TransformerMixin):
    def __init__(self, code_column, description_column):
        """
        Custom transformer to fill missing descriptions based on a mapping derived from the data.

        Args:
            code_column (str): The column with codes to map descriptions.
            description_column (str): The column where missing descriptions will be filled.
        """
        self.code_column = code_column
        self.description_column = description_column
        self.description_dict = {}

    def fit(self, X, y=None):
        # Create the mapping dictionary from the training data
        unique_pairs = X[[self.code_column, self.description_column]].drop_duplicates()
        self.description_dict = unique_pairs[
            unique_pairs[self.description_column].notna()
        ].set_index(self.code_column)[self.description_column].to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Fill missing descriptions using the mapping
        X_copy[self.description_column] = X_copy[self.description_column].fillna(
            X_copy[self.code_column].map(self.description_dict)
        )
        return X_copy


class ImputeAccidentDate(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Custom transformer to impute missing accident dates using the median difference
        between assembly and accident dates. Operates specifically on 'Assembly Date'
        and 'Accident Date'.
        """
        self.median_difference = None

    def fit(self, X, y=None):
        # Calculate the median difference between 'Assembly Date' and 'Accident Date'
        valid_cases = X.dropna(subset=['Assembly Date', 'Accident Date'])
        valid_cases['Date Difference'] = valid_cases.apply(
            lambda row: (row['Assembly Date'] - row['Accident Date']).days
            if row['Assembly Date'] > row['Accident Date']
            else pd.NaT,
            axis=1
        )
        self.median_difference = valid_cases['Date Difference'].dropna().median()
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Condition: missing 'Accident Date' but 'Assembly Date' is present
        condition = (X_copy['Accident Date'].isna()) & (X_copy['Assembly Date'].notna())
        
        # Impute missing 'Accident Date'
        X_copy.loc[condition, 'Accident Date'] = (
            X_copy.loc[condition, 'Assembly Date'] - pd.to_timedelta(self.median_difference, unit='days')
        )
        
        return X_copy

class ImputeAgeAtInjury(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Custom transformer to impute missing 'Age at Injury' based on the difference
        between 'Accident Date' and 'Birth Year'.
        """
        pass

    def fit(self, X, y=None):
        # No fitting required, as we're just applying the same logic in transform
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Filter rows where 'Age at Injury' is missing and 'Accident Date' is present
        condition = X_copy['Age at Injury'].isna() & X_copy['Accident Date'].notna()

        # Impute 'Age at Injury' by calculating the difference between 'Accident Date' and 'Birth Year'
        X_copy.loc[condition, 'Age at Injury'] = (
            X_copy.loc[condition, 'Accident Date'].dt.year - X_copy.loc[condition, 'Birth Year']
        )

        return X_copy



class FillNaNValues(BaseEstimator, TransformerMixin):
    def __init__(self, column, fill_value):
        """
        Custom transformer to fill missing values in the specified column.
        
        Args:
            column (str): The column name to fill missing values in.
            fill_value: The value to replace missing values with.
        """
        self.column = column
        self.fill_value = fill_value

    def fit(self, X, y=None):
        # No fitting required for simple constant filling
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Fill missing values in the specified column
        X_copy[self.column] = X_copy[self.column].fillna(self.fill_value)
        return X_copy
    

class FillNaNValuesColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column, fill_value):
        """
        Custom transformer to fill missing values in the specified column.
        
        Args:
            column (str): The column name to fill missing values in.
            fill_value: The value to replace missing values with.
        """
        self.column = column
        self.fill_value = fill_value

    def fit(self, X, y=None):
        # No fitting required for simple constant filling
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Create a new column with the name '<column> + filled'
        filled_column_name = f"{self.column}_Imputed"
        X_copy[filled_column_name] = X_copy[self.column]
        
        # Fill missing values in the new column
        X_copy[filled_column_name] = X_copy[filled_column_name].fillna(self.fill_value)
        
        return X_copy




class ImputeWithUnknownWCIO(BaseEstimator, TransformerMixin):
    def __init__(self, columns_code, columns_desc, default_code=100, default_desc="Unknown"):
        """
        Custom transformer to impute missing rows where all specified columns are NaN.
        
        Args:
            columns_code (list): List of columns containing codes.
            columns_desc (list): List of columns containing descriptions.
            default_code (int): Default value for code columns.
            default_desc (str): Default value for description columns.
        """
        self.columns_code = columns_code
        self.columns_desc = columns_desc
        self.default_code = default_code
        self.default_desc = default_desc
    
    def fit(self, X, y=None):
        # No fitting is necessary for this transformer
        return self
    
    def transform(self, X):
        # Ensure we don't modify the original DataFrame
        X_copy = X.copy()
        
        # Combine code and description columns for the condition
        missing_condition = X_copy[self.columns_code + self.columns_desc].isna().all(axis=1)
        
        # Apply default values for code and description columns
        for col in self.columns_code:
            X_copy.loc[missing_condition, col] = self.default_code
        for col in self.columns_desc:
            X_copy.loc[missing_condition, col] = self.default_desc
        
        return X_copy


class FillMissingDescriptionsWithMapping(BaseEstimator, TransformerMixin):
    def __init__(self, code_column, description_column):
        self.code_column = code_column
        self.description_column = description_column
    
    def fit(self, X, y=None):
        # Create a DataFrame with unique WCIO Part Of Body Codes and their descriptions (excluding missing descriptions)
        body_code_description = X[[self.code_column, self.description_column]].drop_duplicates()
        body_code_description = body_code_description[body_code_description[self.description_column].notna()]
        
        # Create a dictionary to map codes to descriptions
        self.mapping_dict = body_code_description.set_index(self.code_column)[self.description_column].to_dict()
        return self
    
    def transform(self, X):
        # Create a copy to avoid modifying the original data
        X_copy = X.copy()
        
        # Fill missing descriptions based on the mapping
        X_copy[self.description_column] = X_copy.apply(
            lambda row: self.mapping_dict.get(row[self.code_column], row[self.description_column])
            if pd.isna(row[self.description_column]) else row[self.description_column],
            axis=1
        )
        
        return X_copy

class ImputeUsingModeAfterGrouping(BaseEstimator, TransformerMixin):
    def __init__(self, grouping_column, column_to_impute):
        self.grouping_column = grouping_column  # The column to use for grouping
        self.column_to_impute = column_to_impute  # The column to impute

    def fit(self, X, y=None):
        # Calculate the most frequent (mode) value of the column_to_impute based on the grouping_column
        self.modes = X.groupby(self.grouping_column)[self.column_to_impute].apply(lambda x: x.mode()[0])
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i, row in X_copy.iterrows():
            if pd.isna(row[self.column_to_impute]):  # If the column_to_impute is NaN
                grouping_value = row[self.grouping_column]
                # Fill the missing column_to_impute with the mode of the corresponding group
                if grouping_value in self.modes:
                    X_copy.at[i, self.column_to_impute] = self.modes[grouping_value]
        return X_copy

class ImputeUsingModeAfterGroupingColumn(BaseEstimator, TransformerMixin):
    def __init__(self, grouping_column, column_to_impute):
        self.grouping_column = grouping_column  # The column to use for grouping
        self.column_to_impute = column_to_impute  # The column to impute

    def fit(self, X, y=None):
        # Calculate the most frequent (mode) value of the column_to_impute based on the grouping_column
        self.modes = X.groupby(self.grouping_column)[self.column_to_impute].apply(lambda x: x.mode()[0])
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Create a new column with the name '<column_to_impute> + imputed'
        imputed_column_name = f"{self.column_to_impute}_Imputed"
        X_copy[imputed_column_name] = X_copy[self.column_to_impute]
        
        # Impute missing values
        for i, row in X_copy.iterrows():
            if pd.isna(row[self.column_to_impute]):  # If the column_to_impute is NaN
                grouping_value = row[self.grouping_column]
                # Fill the missing column_to_impute with the mode of the corresponding group
                if grouping_value in self.modes:
                    X_copy.at[i, imputed_column_name] = self.modes[grouping_value]
        
        return X_copy 


# class ImputeC2Date(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.avg_diff = None

#     def fit(self, X, y=None):
#         X = X.copy()
#         X['C-2 Date'] = pd.to_datetime(X['C-2 Date'], errors='coerce')
#         X['Accident Date'] = pd.to_datetime(X['Accident Date'], errors='coerce')

#         # Calculate the difference between 'Accident Date' and 'C-2 Date' for non-null C-2 Dates
#         difference_between_accident_c2 = (X['C-2 Date'] - X['Accident Date']).dropna()
        
#         # Compute the average difference
#         self.avg_diff = difference_between_accident_c2.mean()
#         return self

#     def transform(self, X):
#         X = X.copy()
#         X['C-2 Date'] = pd.to_datetime(X['C-2 Date'], errors='coerce')
#         X['Accident Date'] = pd.to_datetime(X['Accident Date'], errors='coerce')
        
#         # Impute only missing 'C-2 Date' using Accident date + average difference
#         X.loc[X['C-2 Date'].isna(), 'C-2 Date'] = X['Accident Date'] + self.avg_diff
#         X['C-2 Date'] = pd.to_datetime(X['C-2 Date'])
        
#         return X
    


class ImputeC2Date(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.avg_diff = None

    def fit(self, X, y=None):
        X = X.copy()
        X['C-2 Date'] = pd.to_datetime(X['C-2 Date'], errors='coerce')
        X['Accident Date'] = pd.to_datetime(X['Accident Date'], errors='coerce')

        # Calculate the difference between 'Accident Date' and 'C-2 Date' for non-null C-2 Dates
        difference_between_accident_c2 = (X['C-2 Date'] - X['Accident Date']).dropna()
        
        # Compute the average difference
        self.avg_diff = difference_between_accident_c2.mean()
        return self

    def transform(self, X):
        X = X.copy()
        X['C-2 Date'] = pd.to_datetime(X['C-2 Date'], errors='coerce')
        X['Accident Date'] = pd.to_datetime(X['Accident Date'], errors='coerce')
        
        # Create a new column name for imputed 'C-2 Date'
        imputed_column_name = 'C-2 Date_Imputed'
        
        # Copy 'C-2 Date' to the new column
        X[imputed_column_name] = X['C-2 Date']
        
        # Impute only missing 'C-2 Date' using Accident date + average difference
        X.loc[X['C-2 Date'].isna(), imputed_column_name] = X['Accident Date'] + self.avg_diff
        
        return X
