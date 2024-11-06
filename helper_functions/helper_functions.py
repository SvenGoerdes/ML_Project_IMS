import pandas as pd



# This notebook contains helper funcitons that I import in other notebooks to make the code more readable and modular

def impute_with_mode(df: pd.DataFrame, target_column: str, group_column: str, unknown_values=['U', 'X']) -> pd.DataFrame: 
    """
    Imputes missing (NaN) or specified unknown values in a target column based on the most common value (mode) 
    within each group defined by another column.

    Parameters:
    - df: DataFrame containing the data.
    - target_column: The column in which to impute missing/unknown values.
    - group_column: The column to group by for determining the mode for imputation.
    - unknown_values: List of values to be considered as 'unknown' (default is ['Unknown']).
                      NaN values in the target column are also included and will be imputed.

    Returns:
    - DataFrame with imputed values in the target column. If the mode is empty, pd
    """
    # Standardize missing values in the target column by replacing all unknown values with NaN
    df[target_column] = df[target_column].replace(unknown_values, pd.NA)

    # Calculate the mode (most common value) of the target column for each group | if mode is empty, use pd.NA
    mode_by_group = df.groupby(group_column)[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)

    overall_mode = df[target_column].mode().iloc[0]

    # Impute missing values based on the mode of each group
    df[target_column] = df.apply(
        lambda row: mode_by_group[row[group_column]] if pd.isna(row[target_column]) else row[target_column],
        axis=1
    )

    # Impute if the mode is empty for a certain group with overall mode
    if df[target_column].isna().sum() > 0:
        print(f"Imputing {df[target_column].isna().sum()} missing values with overall mode: {overall_mode}")
        df[target_column] = df[target_column].fillna(overall_mode)

    return df


