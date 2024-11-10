import pandas as pd
from category_encoders import TargetEncoder
import numpy as np

def missing_data(df):
    """
    Gives the count and percentage of missing values for each column in a DataFrame
    """    
    # Number of missing values in each column
    missing_count = df.isnull().sum()
    
    # Percentage of missing values for each column
    missing_percentage = ((missing_count / df.shape[0]) * 100).round(2)
    
    missing_data = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_percentage
    })
    
    # Show only columns with missing values
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    # Sort in descending order
    missing_data = missing_data.sort_values(by='Missing Count', ascending=False)
    return missing_data

def multiple_unique_values(df,group_column, unique_column):
    """
    Identifies `group_column` that have more than one unique value

    """
    # Group by group_column and count unique values in the unique_column
    unique_count = df.groupby(group_column)[unique_column].nunique()
    
    # Filter for cases with more than one unique value
    multiple_types = unique_count[unique_count > 1]
    
    print(f"{group_column} with more than one unique value in {unique_column}:")
    for group in multiple_types.index:
        unique_values = df[df[group_column] == group][unique_column].unique()
        print(f"{group}: {unique_values}")

def invalid_entries(df, date_1, date_2):
    """
    Identifies invalid entries where the first date is greater than the second date
    """
    # Validate the dates
    valid_dates = df[date_1] <= df[date_2]
    
    # Find invalid entries where the dates are not valid and the columns are not null
    invalid = df[~valid_dates & df[date_1].notna() & df[date_2].notna()]
    
    # Print the number of invalid rows
    print(f"Number of invalid entries in {date_1} vs {date_2}: {invalid.shape[0]}")
    
    return invalid

def impute_dates_with_difference(df, target_column, reference_column, condition_column, difference_in_days):
    """
    Imputes values in a target date column where a specified condition is met, using a given
    difference in days.

    """
    # Define a timedelta based on the median difference in days
    timedelta = pd.Timedelta(days=difference_in_days)

    # Define the condition for rows to impute
    condition = (df[condition_column] > df[target_column]) & df[reference_column].notna() & df[target_column].notna()
    
    # Apply the imputation
    df.loc[condition, target_column] = df.loc[condition, reference_column] + timedelta
    
    return df


def impute_prop(values, proportions, X, column_name):
    # Calculate the number of missing values
    missing_count = X[column_name].isnull().sum()
    
    # Generate imputed values based on proportions
    imputed_values = np.random.choice(values, size=missing_count, p=proportions)
    
    # Fill in the missing values directly
    X.loc[X[column_name].isnull(), column_name] = imputed_values
    
    return X

def impute_with(df: pd.DataFrame, target_column: str, group_column = None, unknown_values=['Unknown'], reference_df=None, metric = 'mode'):
    """
    Imputes missing (NaN) or specified unknown values in a target column based on the most common value (mode) 
    within each group defined by another column. If a group has no mode, it falls back to the overall mode 
    calculated from the reference DataFrame or df if no reference is provided.

    Parameters:
    - df: DataFrame containing the data to be imputed.
    - target_column: The column in which to impute missing/unknown values.
    - group_column: The column to group by for determining the mode for imputation.
    - unknown_values: Optional List of values to be considered as 'unknown' (default is ['Unknown']).
                      NaN values in the target column are also included and will be imputed.
    - reference_df: Optional DataFrame to calculate modes from (e.g., training set for validation imputation).
                    If None, modes are calculated from df.

    Returns:
    - DataFrame with imputed values in the target column.
    """
    # Standardize missing values in the target column by replacing all unknown values with NaN
    df[target_column] = df[target_column].replace(unknown_values, pd.NA)
    
    # Use reference_df for calculating modes if provided; otherwise, use df
    mode_source_df = reference_df if reference_df is not None else df

    if group_column is not None:
        # Calculate the mode for each group in the group_column from the reference DataFrame

        if metric == 'mode':
            mode_by_group = mode_source_df.groupby(group_column)[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        elif metric == 'mean':
            mode_by_group = mode_source_df.groupby(group_column)[target_column].agg(lambda x: x.mean().iloc[0] if not x.mean().empty else pd.NA)
        elif metric == 'median':
            mode_by_group = mode_source_df.groupby(group_column)[target_column].agg(lambda x: x.median().iloc[0] if not x.median().empty else pd.NA)


        # Impute missing values based on the mode/mean/median of each group, with a fallback to the overall mode
        df[target_column] = df.apply(
            lambda row: mode_by_group[row[group_column]] if pd.isna(row[target_column]) else row[target_column],
            axis=1
        )

    # Calculate the overall mode of the target column in the reference DataFrame as a fallback
    if metric == 'mode':
        overall_mode = mode_source_df[target_column].mode().iloc[0] if not mode_source_df[target_column].mode().empty else pd.NA
    elif metric == 'mean':
        overall_mode = mode_source_df[target_column].mean().iloc[0] if not mode_source_df[target_column].mean().empty else pd.NA
    elif metric == 'median':
        overall_mode = mode_source_df[target_column].median().iloc[0] if not mode_source_df[target_column].median().empty else pd.NA



    # Replace any remaining NaN values (where group mode was NaN) with the overall mode
    df[target_column] = df[target_column].fillna(overall_mode)

    return df

# write a function that takes two pandas datetime columns and returns the difference in days
def days_between(df, start_date_col, end_date_col):
    """
    Calculates the difference in days between two datetime columns in a DataFrame.

    Parameters:
    - df: DataFrame containing the datetime columns.
    - start_date_col: The column containing the start date.
    - end_date_col: The column containing the end date.

    Returns:
    - Series containing the difference in days between the two columns.
    """
    return (df[end_date_col] - df[start_date_col]).dt.days

def target_encode_multiclass(X_train_df, X_val_df, y_train,  feature_col, target_col):
    """
    Applies target encoding to a categorical feature for each class in a multi-class target variable.

    Parameters:
    - X_train (pd.DataFrame): The training feature set.
    - y_train (pd.DataFrame or pd.Series): The training target variable.
    - feature_col (str): The name of the categorical feature column to encode.
    - target_col (str): The name of the target variable column.

    Returns:
    - pd.DataFrame: The original training set concatenated with the encoded features.
    """


    # Ensure y_train is a DataFrame
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()

    # Initialize an empty DataFrame to store encoded features
    encoded_features_train = pd.DataFrame(index=X_train_df.index)
    encoded_features_val = pd.DataFrame(index=X_val_df.index)
    # Loop through each unique category in the targetf
    for outcome in y_train[target_col].unique():
        # Binary target for the current outcome
        y_binary = (y_train[target_col] == outcome).astype(int)

        # Apply target encoding for this outcome
        encoder = TargetEncoder(cols=[feature_col])

        # Fit and transform the training set,
        encoded_column_train = encoder.fit_transform(X_train_df[[feature_col]], y_binary)

        # then transform the validation set using the y_train binary target to avoid data leakage
        encoded_column_val = encoder.transform(X_val_df[[feature_col]])

        # Rename and add the encoded column to the DataFrame
        encoded_features_train[f'{feature_col}_encoded_{outcome}'] = encoded_column_train[feature_col]
        encoded_features_val[f'{feature_col}_encoded_{outcome}'] = encoded_column_val[feature_col]

    # Concatenate the encoded columns with the original training set
    X_train_encoded = pd.concat([X_train_df.reset_index(drop=True), encoded_features_train.reset_index(drop=True)], axis=1)
    X_val_encoded = pd.concat([X_val_df.reset_index(drop=True), encoded_features_val.reset_index(drop=True)], axis=1)

    # return both the training and validation set
    return X_train_encoded, X_val_encoded

def remove_outliers_iqr(df, columns, threshold=1.5):
    df_filtered = df.copy()

    # Calculate Q1, Q3, and IQR for each column
    for col in columns:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1

        # Lower and upper bound for each column
        low_bound = Q1 - threshold * IQR
        up_bound = Q3 + threshold * IQR

        # Rows in beginning vs end
        initial_count = df_filtered.shape[0]
        df_filtered = df_filtered[(df_filtered[col] >= low_bound) & (df_filtered[col] <= up_bound)]
        final_count = df_filtered.shape[0]

        # Calculate and print the percentage of outliers removed
        outliers_removed = initial_count - final_count
        outlier_percentage = (outliers_removed / initial_count) * 100
        print(f'number removed:{outliers_removed}')

    return low_bound, up_bound, df_filtered

