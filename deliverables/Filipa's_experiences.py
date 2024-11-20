#1. IMPORT THE NEEDED LIBRARIES
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree #decision tree
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


#2. IMPORT OUR DATA
WCB = pd.read_csv('project_data/train_data.csv', delimiter=',', dtype={'Zip Code': str}, index_col='Claim Identifier')
X_test = pd.read_csv('project_data/test_data.csv', delimiter=',', dtype={'Zip Code': str}, index_col='Claim Identifier')

#select just these columns for now
selected_columns = ['Age at Injury', 'Alternative Dispute Resolution', 'Attorney/Representative', 
                    'Average Weekly Wage', 'Birth Year', 'Carrier Type', 'Claim Injury Type', 
                    'County of Injury', 'COVID-19 Indicator', 'District Name', 'Gender', 
                    'IME-4 Count', 'Industry Code', 'Medical Fee Region', 
                    'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code', 
                    'WCIO Part Of Body Code', 'Number of Dependents']

WCB = WCB[selected_columns]

#3. EXPLORE THE DATA (CONTINUE...)
# A=WCB.groupby('Claim Injury Type')['Age at Injury'].mean()
# WCB['Claim Injury Type'].value_counts()
# WCB.drop(columns = 'Claim Injury Type').corr(method = 'spearman')


# 4. MODIFY THE DATA
WCB = WCB.dropna(subset=['Claim Injury Type'])  # Remove rows where 'Claim Injury Type' is missing
X = WCB.drop(columns=['Claim Injury Type'])
y = WCB[['Claim Injury Type']]

# Create the imputer for numeric and categorical columns
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Imputer for numeric features (using median) and categorical features (using mode)
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                 test_size=0.3,
                                                 shuffle=True,
                                                 random_state=0,
                                                 stratify=y)

# Pipeline (Imputation + OneHotEncoding + Classification)
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_imputer, numeric_features),  # Impute numeric columns with median
        ('cat', categorical_imputer, categorical_features),  # Impute categorical columns with mode
        ('onehot', OneHotEncoder(), categorical_features)  # One-hot encode categorical columns
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

# Create the full pipeline with the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(criterion='entropy', random_state=42))  # Using ID3 (entropy)
])

# 5. MODELLING
pipeline.fit(X_train, y_train)
predictions_train = pipeline.predict(X_train)
predictions_val = pipeline.predict(X_val)

# 6. ASSESS
train_accuracy = pipeline.score(X_train, y_train)
val_accuracy = pipeline.score(X_val, y_val)

print(f"Accuracy (Train Set): {train_accuracy}")
print(f"Accuracy (Val Set): {val_accuracy}")
print("Confusion Matrix - Train:\n", confusion_matrix(y_train, predictions_train))
print("Confusion Matrix - Val:\n", confusion_matrix(y_val, predictions_val))

# 9. DEPLOY
# X_test['Claim Injury Type'] = model_dt.predict(X_test)
# X_test['Claim Injury Type'].to_csv('predictions.csv')

# Decision Tree graph
plt.figure(figsize=(10, 6))
plot_tree(pipeline.named_steps['classifier'], feature_names=X.columns, class_names=pipeline.named_steps['classifier'].classes_, filled=True)
plt.show()





#	Accident Date	Age at Injury	Alternative Dispute Resolution	Assembly Date	
# Attorney/Representative	Average Weekly Wage	Birth Year	C-2 Date	C-3 Date	
# Carrier Name	Carrier Type	Claim Identifier	Claim Injury Type	County of Injury	
# COVID-19 Indicator	District Name	First Hearing Date	Gender	IME-4 Count	
# Industry Code	Industry Code Description	Medical Fee Region	OIICS Nature of Injury Description
# WCIO Cause of Injury Code	WCIO Cause of Injury Description	WCIO Nature of Injury Code	
# WCIO Nature of Injury Description	WCIO Part Of Body Code	WCIO Part Of Body Description	
# Zip Code	Agreement Reached	WCB Decision	Number of Dependents


#MISSING: Accident Date, Assembly Date, C-2 Date, C-3 Date, Carrier Name, First Hearing Date,
# Claim Identifier,
#Industry Code Description, WCIO Cause of Injury Description, WCIO Nature of Injury Description,
#  WCIO Part Of Body Description, Zip Code