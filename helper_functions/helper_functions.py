import pandas as pd
from category_encoders import TargetEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


# NEW UPDATE TO FUNCTION: TO SET C-2 TO NAN
def handle_outliers_date(X, dataset_type, lower_bound=None, 
                                  date_column='Accident Date', c2_column='C-2 Date'):
    X_copy = X.copy()

    if dataset_type == 'train':
        # Calculate lower bound for training data
        lower_bound = X_copy[date_column].quantile(0.01)
        
        # Drop rows with accident dates below the lower bound
        initial_len = len(X_copy)
        X_copy = X_copy[X_copy[date_column] >= lower_bound]
        print(f"Dropped {initial_len - len(X_copy)} rows from training data based on lower bound.")
        print(f"Lower bound for accident date: {lower_bound}.")
        
        return X_copy, lower_bound
    
    elif dataset_type in ['val', 'test']:
        if lower_bound is None:
            raise ValueError(f"Lower bound must be provided for {dataset_type} datasets.")
        
        # Set C-2 Date to NaT where Accident Date is below the lower bound
        below_bound_mask = X_copy[date_column] < lower_bound
        X_copy.loc[below_bound_mask, c2_column] = pd.NaT

        # Cap Accident Date to the lower bound
        X_copy[date_column] = X_copy[date_column].clip(lower=lower_bound)
        return X_copy

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

def log_remove_outliers_iqr(df, column, threshold=1.5):
    df_filtered = df.copy()

    # Calculate Q1, Q3, and IQR for the specified column
    Q1 = df_filtered[column].quantile(0.25)
    Q3 = df_filtered[column].quantile(0.75)
    IQR = Q3 - Q1

    # Lower and upper bound for the column
    low_bound = Q1 - threshold * IQR
    up_bound = Q3 + threshold * IQR

    # Rows in beginning vs end
    initial_count = df_filtered.shape[0]
    df_filtered = df_filtered[(df_filtered[column] >= low_bound) & (df_filtered[column] <= up_bound)]
    final_count = df_filtered.shape[0]

    # Calculate and print the percentage of outliers removed
    # outliers_removed = initial_count - final_count
    # outlier_percentage = (outliers_removed / initial_count) * 100

    print(f'Lower bound: {np.expm1(low_bound)}') # exponential + 1
    print(f'Upper bound: {np.expm1(up_bound)}')
    # print(f'Number removed: {outliers_removed}')

    # Return the bounds and the filtered dataframe
    return np.expm1(low_bound), np.expm1(up_bound), df_filtered

def remove_outliers_iqr(df, column, threshold=1.5):
    df_filtered = df.copy()

    # Calculate Q1, Q3, and IQR for the specified column
    Q1 = df_filtered[column].quantile(0.25)
    Q3 = df_filtered[column].quantile(0.75)
    IQR = Q3 - Q1

    # Lower and upper bound for the column
    low_bound = Q1 - threshold * IQR
    up_bound = Q3 + threshold * IQR

    # Rows in beginning vs end
    initial_count = df_filtered.shape[0]
    df_filtered = df_filtered[(df_filtered[column] >= low_bound) & (df_filtered[column] <= up_bound)]
    final_count = df_filtered.shape[0]

    # outliers_removed = initial_count - final_count
    # outlier_percentage = (outliers_removed / initial_count) * 100

    print(f'Lower bound: {low_bound}') 
    print(f'Upper bound: {up_bound}')
    # print(f'Number of outliers in {column}: {outliers_removed}')

    # Return the bounds and the filtered dataframe
    return low_bound, up_bound, df_filtered

def iqr_date(df, date_column):
    """
    Calculate IQR for a date column and identify outliers.
    
    Parameters:
    - df: DataFrame.
    - date_column: Name of the column that we want to check outliers for.  
    Returns:
    - lower_bound_date: The lower bound date for identifying outliers.
    - upper_bound_date: The upper bound date for identifying outliers.
    """
    #Need to convert date to numeric stamps
    timestamps = df[date_column].dropna().apply(lambda x: x.timestamp())

    Q1 = timestamps.quantile(0.25)
    Q3 = timestamps.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound_timestamp = Q1 - 1.5 * IQR
    upper_bound_timestamp = Q3 + 1.5 * IQR

    lower_bound_date = pd.to_datetime(lower_bound_timestamp, unit='s')
    upper_bound_date = pd.to_datetime(upper_bound_timestamp, unit='s')

    outlier_count = ((df[date_column] < lower_bound_date) | (df[date_column] > upper_bound_date)).sum()

    print(f"Lower Bound Date for {date_column}: {lower_bound_date}")
    print(f"Upper Bound Date for {date_column}: {upper_bound_date}")
    print(f"Number of outliers in {date_column}: {outlier_count}")

    return lower_bound_date, upper_bound_date



def handle_outliers_age_birth(X, dataset_type, age_column='Age at Injury', birth_year_column='Birth Year', 
                    lower_bound=14, z_score_threshold=3, train_stats=None):
    """
    Handle outliers by either dropping (train) or setting to NaN (val/test) based on dataset type.

    Parameters:
    - X: DataFrame to process.
    - dataset_type: 'train', 'val', or 'test'.
    - age_column: Name of the 'Age at Injury' column.
    - birth_year_column: Name of the 'Birth Year' column.
    - lower_bound: Minimum valid age.
    - z_score_threshold: Z-score threshold for upper bound.
    - train_stats: Dictionary containing mean and std from the training set (optional for train).

    Returns:
    - Transformed DataFrame.
    - Dictionary containing the mean and std from the training set (if dataset_type='train').
    """
    if dataset_type == 'train':
        # Compute mean and std for training data
        age_data = X[age_column].dropna()
        mean = age_data.mean()
        std = age_data.std()
    elif train_stats is not None:
        # Use mean and std from training data
        mean = train_stats['mean']
        std = train_stats['std']
    else:
        raise ValueError("X_train stats must be provided for validation or test datasets.")

    # Calculate the upper bound for outliers
    upper_bound = mean + z_score_threshold * std

    X_copy = X.copy()
    
    # Calculate z-scores for 'Age at Injury'
    z_scores = (X_copy[age_column] - mean) / std

    # Initialize counter for outliers
    num_outliers = 0

    # Detect outliers (both for training and non-training datasets)
    outlier_condition = (z_scores > z_score_threshold) | (X_copy[age_column] < lower_bound)
    num_outliers = outlier_condition.sum()

    if dataset_type == 'train':
        # Drop outliers for training
        X_copy.loc[outlier_condition, age_column] = np.nan
    else:
        # Set outliers to NaN for validation/test (instead of capping)
        X_copy.loc[outlier_condition, age_column] = np.nan

    # Set 'Birth Year' to NaN only where 'Age at Injury' is NaN due to outliers
    X_copy.loc[outlier_condition, birth_year_column] = np.nan

    # Print information
    print(f"Dataset: {dataset_type}")
    print(f"Number of outliers detected: {num_outliers}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound:.2f}")
    print("-" * 40)

    if dataset_type == 'train':
        return X_copy, {'mean': mean, 'std': std}
    else:
        return X_copy
    


    
def process_birth_year(df, accident_date_col='Accident Date', birth_year_col='Birth Year', age_col='Calculated Age at Injury'):
    """
    Process the age at injury and handle invalid values in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        accident_date_col (str): Name of the 'Accident Date' column.
        birth_year_col (str): Name of the 'Birth Year' column.
        age_col (str): Name of the column to store the calculated age at injury.
        drop_accident_year (bool): Whether to drop the 'Accident Year' column after processing.

    Returns:
        pd.DataFrame: Transformed DataFrame with calculated age and invalid rows set to NaN.
    """
    # Ensure 'Accident Date' is datetime
    df[accident_date_col] = pd.to_datetime(df[accident_date_col], errors='coerce')
    
    # Extract year from 'Accident Date'
    df['Accident Year'] = df[accident_date_col].dt.year
    
    # Calculate age at injury
    df[age_col] = df['Accident Year'] - df[birth_year_col]
    
    # Set invalid ages to NaN
    invalid_condition = (df[age_col] < 14) | (df[age_col] > 83)
    df.loc[invalid_condition, [age_col, birth_year_col]] = np.nan
    
    df.drop(columns=['Accident Year','Calculated Age at Injury'], inplace=True)    
    
    # Print summary
    print(f"Number of rows where {birth_year_col} set to NaN: {invalid_condition.sum()}")
    
    return df



def remove_ime4_outliers_train(df, column, threshold=20):
    # Create a copy of the input DataFrame
    df_cleaned = df.copy()
    
    # Count how many rows have values greater than the threshold
    outliers_count = (df_cleaned[column] > threshold).sum()
    
    # Remove the outliers, but keep rows where the column is NaN
    df_cleaned = df_cleaned[(df_cleaned[column] <= threshold) | df_cleaned[column].isna()]
    
    # Print the number of removed rows
    print(f"Removed: {outliers_count} outliers")
    
    return df_cleaned


def cap_ime4_outliers(df, column, threshold=20):
    df[column] = df[column].clip(upper=threshold)
    return df



def handle_outliers_with_log_iqr(train_df, val_df, test_df, column='Average Weekly Wage'):
    """
    Handles outliers in a specified column using log transformation and IQR.
    Removes non-zero outliers in the training set and caps non-zero values in validation and test sets.
    
    Parameters:
        train_df (pd.DataFrame): Training dataset
        val_df (pd.DataFrame): Validation dataset
        test_df (pd.DataFrame): Test dataset
        column (str): Column name for outlier handling
    
    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Modified training, validation, and test datasets
    """
    # Copy datasets to avoid modifying the originals
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    # Filter out zero values from the training set for outlier detection
    train_non_zero = train_df[train_df[column] > 0].copy()
    
    # Apply log transformation to the non-zero values
    train_non_zero['log_' + column] = np.log(train_non_zero[column])
    
    # Calculate IQR on the log-transformed values
    Q1 = train_non_zero['log_' + column].quantile(0.25)
    Q3 = train_non_zero['log_' + column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds in the original scale
    lower_bound = np.exp(Q1 - 1.5 * IQR)
    upper_bound = np.exp(Q3 + 1.5 * IQR)
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    
    # Remove non-zero outliers from the training set, but keep NaN values
    initial_train_size = len(train_df)
    train_df = train_df[(train_df[column].isna()) | (train_df[column] == 0) | ((train_df[column] >= lower_bound) & (train_df[column] <= upper_bound))]
    outliers_removed = initial_train_size - len(train_df)
    print(f"Removed {outliers_removed} outliers from the training set.")
    
    # Cap non-zero values in the validation and test datasets
    for df in [val_df, test_df]:
        non_zero_mask = df[column] > 0  # Identify non-zero elements
        df.loc[non_zero_mask, column] = df.loc[non_zero_mask, column].clip(lower=lower_bound, upper=upper_bound)
    
    # Drop the temporary column from the filtered training dataset
    train_non_zero.drop(columns=['log_' + column], inplace=True)
    
    return train_df, val_df, test_df


def plot_accident_date_distributions_separate(datasets, labels, column_name='Accident Date'):
    """
    Plot the distributions of Accident Date for multiple datasets in separate subplots.

    Parameters:
    - datasets: List of DataFrames.
    - labels: List of labels for the datasets.
    - column_name: Name of the accident date column to plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for ax, data, label in zip(axes, datasets, labels):
        sns.kdeplot(data[column_name], ax=ax, fill=True, alpha=0.5)
        ax.set_title(f'{label} Distribution')
        ax.set_xlabel('Accident Date')
        ax.set_ylabel('Density')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Accident Date Distributions Before and After Processing')
    plt.tight_layout()
    plt.show()
