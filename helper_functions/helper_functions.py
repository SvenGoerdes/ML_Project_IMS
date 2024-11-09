import pandas as pd

def impute_with(df: pd.DataFrame, target_column: str, group_column = None, unknown_values=['Unknown'], reference_df=None, metric = 'mode'):
    """
    Imputes missing (NaN) or specified unknown values in a target column based on the most common value (mode) 
    within each group defined by another column. If a group has no mode, it falls back to the overall mode 
    calculated from the reference DataFrame or df if no reference is provided.

    Parameters:
    - df: DataFrame containing the data to be imputed.
    - target_column: The column in which to impute missing/unknown values.
    - group_column: The column to group by for determining the mode for imputation.
    - unknown_values: List of values to be considered as 'unknown' (default is ['Unknown']).
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
