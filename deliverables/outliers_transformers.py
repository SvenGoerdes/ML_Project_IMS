import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DateCapLower(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, lower_percentile=0.01):
        self.column_name = column_name
        self.lower_percentile = lower_percentile

    def fit(self, X, y=None):
        # Calculate the lower bound based on percentiles for the specific column
        self.lower_bound_ = X[self.column_name].quantile(self.lower_percentile)
        return self

    def transform(self, X):
        # Create a copy of the DataFrame to avoid modifying the original data
        X_capped = X.copy()

        # Cap the specific column based on the calculated lower bound
        X_capped[self.column_name] = X_capped[self.column_name].clip(lower=self.lower_bound_)
        
        return X_capped

    
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class OutlierAgeAtInjuryAndBirthYear(BaseEstimator, TransformerMixin):
    def __init__(self, age_column='Age at Injury', birth_year_column='Birth Year', lower_bound=14, z_score_threshold=3):
        """
        Transformer to impute 'Age at Injury' and 'Birth Year' based on z-score for the upper bound and a fixed lower bound.

        Parameters:
        - age_column (str): Column name for Age at Injury.
        - birth_year_column (str): Column name for Birth Year.
        - lower_bound (int): Minimum valid value for Age at Injury.
        - z_score_threshold (float): Z-score threshold for identifying upper bound outliers.
        """
        self.age_column = age_column
        self.birth_year_column = birth_year_column
        self.lower_bound = lower_bound
        self.z_score_threshold = z_score_threshold

    def fit(self, X, y=None):
        """
        Calculate mean and std for z-score calculation.

        Parameters:
        - X (DataFrame): Input data.

        Returns:
        - self
        """
        age_data = X[self.age_column].dropna()
        self.mean = age_data.mean()
        self.std = age_data.std()
        return self

    def transform(self, X):
        """
        Apply the imputation based on z-score for the upper bound and fixed lower bound.

        Parameters:
        - X (DataFrame): Input data.

        Returns:
        - X_copy (DataFrame): Transformed data.
        """
        X_copy = X.copy()

        # Calculate z-scores for 'Age at Injury'
        z_scores = (X_copy[self.age_column] - self.mean) / self.std

        # Identify rows to set as NaN
        outlier_condition = (z_scores > self.z_score_threshold) | (X_copy[self.age_column] < self.lower_bound)
        
        # Set 'Age at Injury' to NaN for outliers
        X_copy.loc[outlier_condition, self.age_column] = np.nan
        
        # Set 'Birth Year' to NaN where 'Age at Injury' is NaN
        X_copy.loc[X_copy[self.age_column].isna(), self.birth_year_column] = np.nan
        
        return X_copy
