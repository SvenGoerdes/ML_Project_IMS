from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, OneHotEncoder
import pandas as pd
import numpy as np

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, binary_columns):
        self.binary_columns = binary_columns
        self.encodings = {}  # Store the mapping for each column

    # Fit method creates a dictionary which defines the mapping. 
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame
        for col in self.binary_columns:
            unique_values = sorted(X[col].unique())

            if len(unique_values) != 2:
                raise ValueError(f"Column '{col}' does not have exactly two unique values.")

            X[f'{col}_binary'] = X[col].map({unique_values[0]: 0, unique_values[1]: 1})

        return X

class MultipleTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                # target_columns
                feature_column):
        # self.target_columns = target_columns
        self.feature_column = feature_column
        self.target_encoders = {}  # Store the target encoders for each column 

    def fit(self, X, y):

        for unique_value in y.unique():
            
            # for every unique value there needs to be one new column
            y_binary = (y == unique_value).astype(int)

            # create a target encoder for each unique value of the target column
            target_encoder = TargetEncoder(cols=[self.feature_column])

            # fit the encoder
            target_encoder.fit(X[[self.feature_column]], y_binary)

            # safe fit in a dictionary
            self.target_encoders[unique_value] = target_encoder

        return self
    
    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame
        for unique_value, encoder  in self.target_encoders.items():

            # then transform the validation set using the y_train binary target to avoid data leakage
            encoded_column = encoder.transform(X[[self.feature_column]])

            # Rename and add the encoded column to the DataFrame
            X[f'{self.feature_column}_encoded_{unique_value}'] = encoded_column

        return X
    # X contains the dataframe with the respective column that we want to encode 

class Days_between(BaseEstimator, TransformerMixin):
    def __init__(self, start_col, end_col):
        self.start_col = start_col
        self.end_col = end_col

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame
        X[f'Days_between_{self.end_col}_{self.start_col}'] = (X[self.end_col] - X[self.start_col]).dt.days
        return X


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, dummy_column, drop = 'first'):
        self.dummy_column = dummy_column
        self.drop = drop
        self.encoder = OneHotEncoder(drop = self.drop)

    def fit(self, X, y=None):
        # X = X.copy()  # Avoid modifying the original DataFrame
        self.encoder.fit(X[[self.dummy_column]])
        return self


    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame
        transformed_data = self.encoder.transform(X[[self.dummy_column]])
        transformed_df = pd.DataFrame(
            transformed_data.toarray(),
            columns=self.encoder.get_feature_names_out([self.dummy_column]),
            index = X.index # Keep the original index
        ) 

        # Concatenate along columns without dropping the original index
        X = pd.concat([X, transformed_df], axis=1)
        return X

class ColumnMapper(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, mapping_dict = None, drop_original = True):
        self.mapping_dict = mapping_dict
        self.column_name = column_name
        self.drop_original = drop_original

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame

        if self.drop_original:
            X[self.column_name] = X[self.column_name].map(self.mapping_dict)
    
        else:
            X[f'{self.column_name}_mapped'] = X[self.column_name].map(self.mapping_dict)

        return X
    
class NAIndicatorEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer class for encoding missing values in a column as binary indicators.

    Parameters:
    - column_name: The name of the column to encode.
    - include_zero: Whether to include zero values in the encoding (default is False).

    Returns:
    - Dataframe with a new column indicating missing values in the specified column with 0 or 1.
    """
    def __init__(self, column_name, include_zero = False):
        self.column_name = column_name
        self.include_zero = include_zero

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame

        if self.include_zero:
            X[f'{self.column_name}_nabinary'] = ((X[self.column_name].isna()) | (X[self.column_name] == 0)).astype(int)
        # Encode the column as 1 for not NA and 0 for NA
        else:
            X[f'{self.column_name}_nabinary'] = X[self.column_name].isna().astype(int)
        
        return X
    
class SeasonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column, categorical=True):
        self.date_column = date_column
        self.categorical = categorical
        self.season_dict = {
            1: 'Winter',  # Dec (12), Jan (1), Feb (2)
            2: 'Spring',  # Mar (3), Apr (4), May (5)
            3: 'Summer',  # Jun (6), Jul (7), Aug (8)
            4: 'Fall'     # Sep (9), Oct (10), Nov (11)
        }
    
    def fit(self, X, y=None):
        if self.date_column not in X.columns:
            raise ValueError(f"Column {self.date_column} not found in input data")
        return self

    def get_season(self, month):
        if month == 12:
            return 1  # Winter
        return (month % 12 + 3) // 3

    def transform(self, X):
        X = X.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
            try:
                X[self.date_column] = pd.to_datetime(X[self.date_column])
            except:
                raise ValueError(f"Could not convert {self.date_column} to datetime")
        
        # Calculate season number (1-4)
        X[f'{self.date_column}_Season'] = X[self.date_column].dt.month.apply(self.get_season)
        
        # Convert to categorical if requested
        if self.categorical:
            X[f'{self.date_column}_Season'] = X[f'{self.date_column}_Season'].map(self.season_dict)
        
        return X

class CategorizeIncomeDescriptive(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # No hyperparameters to initialize

    def fit(self, X, y=None):
        # Calculate adjusted quantiles
        self.low_threshold_ = X['Average Weekly Wage'].quantile(0.25)
        self.high_threshold_ = X['Average Weekly Wage'].quantile(0.85)
        return self

    def transform(self, X):
        # Create a copy to avoid altering the original data
        X = X.copy()
        
        # Define conditions and choices for vectorized selection
        conditions = [
            X['Average Weekly Wage'] <= self.low_threshold_,
            (X['Average Weekly Wage'] > self.low_threshold_) & (X['Average Weekly Wage'] <= self.high_threshold_),
            X['Average Weekly Wage'] > self.high_threshold_
        ]
        choices = ['Low Income', 'Middle Class', 'Wealthy']
        
        # Apply the vectorized selection
        X['Income_Category'] = np.select(conditions, choices, default='Unknown')
        return X
    

class NumberBinning(BaseEstimator, TransformerMixin): 
    def __init__(self, init_col_name , column_name, bins = [0, 29, 50, 1000], labels = ['Young Workforce', 'Middle-Age', 'Older']):
        self.initial_column_name = init_col_name
        self.new_column_name = column_name

        self.bins = bins
        self.labels = labels
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        X.loc[:, self.new_column_name] = pd.cut(X[self.initial_column_name], bins=self.bins, labels=self.labels)
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer class for applying a logarithmic transformation to a specified column.
    """

    def __init__(self, column, base = np.e, handle_zeros=True, offset=1.0):
        """
        Initialize the LogTransformer.
        
        Parameters:
        - column: The column to apply the transformation to.
        - base: The logarithmic base (default is natural log).
        - handle_zeros: Whether to handle zeros by adding an offset (default is True).
        - offset: The offset value to add to handle zeros and negatives (default is 1.0).
        """
        self.column = column
        self.base = base
        self.handle_zeros = handle_zeros
        self.offset = offset

    def fit(self, X, y=None):
        """
        Fit method for compatibility with sklearn pipeline. Does not need to do anything.
        
        Parameters:
        - X: DataFrame, input data.
        - y: Ignored, not used.
        
        Returns:
        - self: Fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Apply the log transformation to the specified column. Drops the original column and adds the transformed column with a '_log' suffix.
        
        Parameters:
        - X: DataFrame, input data.
        
        Returns:
        - Transformed DataFrame with the logarithm applied to the specified column.
        """
        # X = pd.DataFrame(X)  # Ensure we are working with a DataFrame
        X = X.copy()

        # print(f"Column stats before transform:")
        # print(f"NaN values: {X[self.column].isna().sum()}")
        # print(f"Negative values: {(X[self.column] < 0).sum()}")
        # print(f"Zero values: {(X[self.column] == 0).sum()}")
        # print(f"Min value: {X[self.column].min()}")
        

        # X[f'{self.column}_log'] = np.log1p(X[self.column])

        # replace values with negative values with 0
        X[self.column] = X[self.column].apply(lambda x: 0 if x < 0 else x)

        if self.handle_zeros:
            self.zero_corrected = X[self.column] + self.offset  # Add offset for zeros and negatives
        
        X[f'{self.column}_log'] = (
            np.log(self.zero_corrected + 1) / np.log(self.base)  # Apply log to the specified column
        )
        
        return X

class frequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        self.frequency_map = {}

    def fit(self, X, y=None):
        self.frequency_map = X[self.column_name].value_counts().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X[f'{self.column_name}_freq'] = X[self.column_name].map(self.frequency_map).fillna(0)
        return X
    

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract day of the week, month, and year 
    from a datetime column in a pandas DataFrame.

    Args:
        date_column (str): The name of the datetime column in the input DataFrame.

    Returns:
        A pandas DataFrame with extracted features.


    """
    def __init__(self, date_column: str):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        if self.date_column not in X.columns:
            raise KeyError(f"The specified column '{self.date_column}' is not in the DataFrame.")

        # Ensure the column is of datetime type
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')

        # Extract features
        X[f'{self.date_column}_day_of_week'] = X[self.date_column].dt.dayofweek  # Monday=0, Sunday=6
        X[f'{self.date_column}month'] = X[self.date_column].dt.month            # January=1, December=12
        X[f'{self.date_column}year'] = X[self.date_column].dt.year

        return X