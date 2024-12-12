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
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame

        # Encode the column as 1 for not NA and 0 for NA
        X[f'{self.column_name}_nabinary'] = X[self.column_name].notna().astype(int)
        
        return X

class SeasonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column ):
        self.date_column = date_column
        self.season_dict = {
            "Spring": 1,
            "Summer": 2,
            "Autumn": 3,
            "Winter": 4
        }
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[f'{self.date_column}_Season'] = X[self.date_column].dt.month % 12 // 3 + 1
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

        # this was for debugging purposes --- there are some rows where c2 date is before accident date in val and test 
        # print(f"Column stats before transform:")
        # print(f"NaN values: {X[self.column].isna().sum()}")
        # print(f"Negative values: {(X[self.column] < 0).sum()}")
        # print(f"Zero values: {(X[self.column] == 0).sum()}")
        # print(f"Min value: {X[self.column].min()}")
        

        # X[f'{self.column}_log'] = np.log1p(X[self.column])

        # replace values with negative values with 0
        self.zero_corrected = X[self.column].apply(lambda x: 0 if x < 0 else x)

        if self.handle_zeros:
            self.zero_corrected = self.zero_corrected + self.offset  # Add offset for zeros and negatives
            # self.zero_corrected = X[self.column] + self.offset  # Add offset for zeros and negatives      # This needs to get fixed here.
        
        X[f'{self.column}_log'] = (
            np.log(self.zero_corrected + 1) / np.log(self.base)  # Apply log to the specified column
        )
        
        return X

