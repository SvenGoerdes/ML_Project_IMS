# -------------- Feature engineering --------------
# The script contains the following feature engineering steps: 
# 
# 1. We add a seasonality, month and day of week feature from the Accident Date column
# 2. We add Principal Component for the columns that contain the industry, and the WCIO codes
# 
# 
# -------------- ------------------ --------------
import pandas as pd
from sklearn.decomposition import PCA
import sys
# create a plot that shows the cumulative variance explained by the components
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# read in project_data
df_train = pd.read_csv('../project_data/X_train_encoded.csv')
df_val = pd.read_csv('../project_data/X_val_encoded.csv')
df_test = pd.read_csv('../project_data/X_test_encoded.csv')


# create new seasonality features
def process_dates(df, date_column):
    
    # check if date column is in the dataframe
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Extract date features
        df[f'{date_column}_Month'] = df[date_column].dt.month
        df[f'{date_column}_DayOfWeek'] = df[date_column].dt.dayofweek # Monday=0, Sunday=6
        
        # Extract season
        df[f'{date_column}_Season'] = df[date_column].dt.month % 12 // 3 + 1  # Seasons: 1-4
        
        # create a dicitonary that maps season to string
        season_dict = {
        1: 'spring',
        2: 'summer',
        3: 'autumn',
        4: 'winter'
        }


        df[f'{date_column}_Season'] = df[f'{date_column}_Season'].map(season_dict)

        # create a dummy variable for the season with int encoding
        df = pd.get_dummies(df, columns=[f'{date_column}_Season'], drop_first=True)
        
        # convert each dummy column into int  
        for col in df.columns[df.columns.str.contains('Accident Date_Season')]:
            df[col] = df[col].astype(int)

        # drop the data_column
        df = df.drop(columns=[date_column])

    return df

# write me a function that creates a new column which is binary. It is zero if the value is missing and 1 if the value is not missing
def create_missing_column(df, column_name):
    df[f'{column_name}_missing'] = df[column_name].isnull().astype(int)
    return df


# apply function to all dataframes
df_train = process_dates(df_train, 'Accident Date')
df_val = process_dates(df_val, 'Accident Date')
df_test = process_dates(df_test, 'Accident Date')

df_train =  create_missing_column(df_train, 'Average Weekly Wage')
df_val =  create_missing_column(df_val, 'Average Weekly Wage')
df_test =  create_missing_column(df_test, 'Average Weekly Wage')

# Principal Component Analysis (PCA)

# If we only have one command line argument we will exit the script if not we will compute PCA
if len(sys.argv) == 1:

    # save data to csv
    df_train.to_csv('../project_data/X_train_encoded_feat.csv', index=False)
    df_val.to_csv('../project_data/X_val_encoded_feat.csv', index=False)
    df_test.to_csv('../project_data/X_test_encoded_feat.csv', index=False)

    # exit script
    sys.exit(0)

# use the following columns for PCA
columns_pca =   ['Industry Code_encoded_5. PPD SCH LOSS',
    'Industry Code_encoded_2. NON-COMP',
    'Industry Code_encoded_3. MED ONLY',
    'Industry Code_encoded_4. TEMPORARY',
    'Industry Code_encoded_1. CANCELLED', 'Industry Code_encoded_8. DEATH',
    'Industry Code_encoded_6. PPD NSL', 'Industry Code_encoded_7. PTD',

    'WCIO Cause of Injury Code_encoded_5. PPD SCH LOSS',
    'WCIO Cause of Injury Code_encoded_2. NON-COMP',
    'WCIO Cause of Injury Code_encoded_3. MED ONLY',
    'WCIO Cause of Injury Code_encoded_4. TEMPORARY',
    'WCIO Cause of Injury Code_encoded_1. CANCELLED',
    'WCIO Cause of Injury Code_encoded_8. DEATH',
    'WCIO Cause of Injury Code_encoded_6. PPD NSL',
    'WCIO Cause of Injury Code_encoded_7. PTD',

    'WCIO Nature of Injury Code_encoded_5. PPD SCH LOSS',
    'WCIO Nature of Injury Code_encoded_2. NON-COMP',
    'WCIO Nature of Injury Code_encoded_3. MED ONLY',
    'WCIO Nature of Injury Code_encoded_4. TEMPORARY',
    'WCIO Nature of Injury Code_encoded_1. CANCELLED',
    'WCIO Nature of Injury Code_encoded_8. DEATH',
    'WCIO Nature of Injury Code_encoded_6. PPD NSL',
    'WCIO Nature of Injury Code_encoded_7. PTD',

    'WCIO Part Of Body Code_encoded_5. PPD SCH LOSS',
    'WCIO Part Of Body Code_encoded_2. NON-COMP',
    'WCIO Part Of Body Code_encoded_3. MED ONLY',
    'WCIO Part Of Body Code_encoded_4. TEMPORARY',
    'WCIO Part Of Body Code_encoded_1. CANCELLED',
    'WCIO Part Of Body Code_encoded_8. DEATH',
    'WCIO Part Of Body Code_encoded_6. PPD NSL',
    'WCIO Part Of Body Code_encoded_7. PTD']

# scale data with min-max scaling before applying PCA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_train[columns_pca])
df_train[columns_pca] = scaler.transform(df_train[columns_pca])
df_val[columns_pca] = scaler.transform(df_val[columns_pca])
df_test[columns_pca] = scaler.transform(df_test[columns_pca])

# perform pca on the data and print out the variance explained by each component
pca = PCA()
pca.fit(df_train[columns_pca])

#  ------- Plot the cumulative variance explained by the components ---- 
sns.set_theme()
fig, ax = plt.subplots(1,2)

ax[0].plot(np.cumsum(pca.explained_variance_ratio_[:15]))
ax[0].set_xlabel('Number of Components')
ax[0].set_ylabel('Variance (%)') #for each component
ax[0].set_title('Explained Variance')

ax[1].plot(pca.explained_variance_ratio_[:15])
ax[1].set_xlabel('Number of Components')
ax[1].set_ylabel('Variance (%)') #for each component
ax[1].set_title('Explained Variance')

plt.tight_layout()


# save the plot to a png file 
plt.savefig('../project_data/pca_variance.png')

#  ----- End of plot ---- 
# You can see the plotfile in the project_data folder 
# We can see that the first 5 components explain over 80% of the variance in the data

# use transform to transform the data for test, val and train 
df_train_pca = pca.transform(df_train[columns_pca])
df_val_pca = pca.transform(df_val[columns_pca])
df_test_pca = pca.transform(df_test[columns_pca])

# add the PCA components to the dataframes
for i in range(5):
    df_train[f'PCA{i+1}'] = df_train_pca[:, i]
    df_val[f'PCA{i+1}'] = df_val_pca[:, i]
    df_test[f'PCA{i+1}'] = df_test_pca[:, i]

# save each df to csv
df_train.to_csv('../project_data/X_train_encoded_pca.csv', index=False)
df_val.to_csv('../project_data/X_val_encoded_pca.csv', index=False)
df_test.to_csv('../project_data/X_test_encoded_pca.csv', index=False)