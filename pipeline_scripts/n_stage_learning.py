import pandas as pd
import numpy as np

# import the model classes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def fp_tn_distribution(prediction, true_values, minority_class_col = 'minority_class', normalize = False):
    """
    Define a function that takes the prediction and the true values and returns the distribution of the target value of the false positives and true negatives

    Args:
        prediction: prediction values
        true_values: true values
        minority_class_col: column that defines the minority class
        normalize: boolean that defines if the values should be normalized
    
    Returns:
        value_count_fp: distribution of the false positives target value 
        value_count_tn: distribution of the true negatives target value
    """

    if normalize:

        # get the false positives and true negatives rows 
        false_positives = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0) & (prediction == 1)] 
        true_negative = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0) & (prediction == 0)] 

        # labels 
        labels = true_values['Claim Injury Type'].unique()
        # divide the value counts 
        true_values_negative = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0)].value_counts()
        
        # count values for every  for the false positives and true negatives
        value_count_fp = false_positives.value_counts().astype(float)
        value_count_tn = true_negative.value_counts().astype(float)


        # if a value is missing in the false positives or true negatives add it with a value of 0
        for value in labels:
            if value not in value_count_fp.index:
                value_count_fp[value] = 0
            if value not in value_count_tn.index:
                value_count_tn[value] = 0


        # divide the value counts by the total value counts
        for value in true_values_negative.index:
            value_count_fp[value] = value_count_fp[value] / true_values_negative[value]
            value_count_tn[value] = value_count_tn[value] / true_values_negative[value]
        

    else:
        false_positives = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0) & (prediction == 1)] 
        true_negative = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0) & (prediction == 0)] 
        value_count_fp = false_positives.value_counts()
        value_count_tn = true_negative.value_counts()


    return value_count_fp, value_count_tn

def model_split(binary_list: list, all_features: list):
    """
    Define a function that takes a list that define the splits for the each node and returns a list of tuples that define the model training in each iteration

    Args:
        binary_list: list of lists that define the binary split for each node
        all_features: list of all features that will be used in the model

    Returns:
        model_iteration_list: list of tuples that define the model training in each iteration/node 
    """

    iteration_features = all_features.copy()
    model_iteration_list = []

    # iterate through each feature combination 
    for binary_target in binary_list:
        
        # we need to get each feature out of the list 
        for feature in binary_target:
            if feature in iteration_features:

                iteration_features.remove(feature)

        # append the tuple of binary_target and remaining features
        i_iteration_binary_train = (binary_target, iteration_features.copy())
        model_iteration_list.append(i_iteration_binary_train)

    return model_iteration_list

def binary_splitting(df: pd.DataFrame, minority_class_list: list, split_col):
    """
    This function takes in a dataframe and returns a binary array where the minority class is 1 and the majority class is 0.
    
    Args:    
        df: the target dataframe with the main target column
        minority_class_list: A list of the minority class
        split_col: The column that you want to split on mostly the main target column

    Returns:
        
        binary_array: A numpy array with the minority class as 1 and the majority class
    """

    df_copy = df.copy()

    if (len(minority_class_list) == 0):
        raise ValueError('minority_class_list has to contain atleast on value')
    elif (len(minority_class_list) == 1):
        binary_array = (df_copy[split_col] == minority_class_list[0]).astype(int).values
    else:

        claim_types = df_copy[split_col].values
        minority_class_array = np.array(minority_class_list)
        binary_array = np.isin(claim_types, minority_class_array).astype(int)
    
    return binary_array

def filter_features(y_train, binary_target, rest_binary_target):

        # filter the features | get all the rows where y_val contains the rest_binary_target values
        y_train_filtered = y_train[y_train['Claim Injury Type'].isin(rest_binary_target + binary_target)].copy()

        y_train_filtered.loc[:, 'binary_target'] = binary_splitting(y_train_filtered, binary_target, 'Claim Injury Type')

        return y_train_filtered

# create a n_stage_learning_model function that will take the model_iteration_list and train the model in each iteration
def n_stage_learning_model(model_iteration_list, X_train, y_train, X_val, y_val, model, **kwargs):
    """
    Define a function that takes a list of tuples that define the model training in each iteration and train the model in each iteration

    Args:
        model_iteration_list: list of tuples that define the model training in each iteration/node 
        X_train: training features
        y_train: training target
        X_test: testing features
        y_test: testing target
        model: model to be trained

    Returns:
        model: trained model
    """

    # create a dictionary to store the models
    model_dict = {}


    model_classes = {
        'Logistic Regression': LogisticRegression,
        'Random Forest': RandomForestClassifier,
        'XGBoost': XGBClassifier,
        'Ridge': RidgeClassifier,
        'KNN': KNeighborsClassifier,
        'Decision Tree': DecisionTreeClassifier,
        'Naive Bayes': GaussianNB

    }

    # keep track of the node
    iter = 1

    model_count = len(model_iteration_list)
    
    # iterate through each feature combination 
    for binary_target, rest_binary_target in model_iteration_list:
        
        # create model instance inside the dictionary
        model_dict[str(binary_target) + '_model_node_' + str(iter)] = model_classes[model](**kwargs)


        # this gives back a dataframe with rows only containing values of the binary_target and rest_binary_target
        # and a column that defines the binary_target values with 0 and 1
        y_train_filtered = filter_features(y_train, binary_target, rest_binary_target)

        X_train_filtered = X_train.loc[y_train_filtered.index]

        # train the model of the node with the filtered features and target values inside the dictionary
        model_dict[str(binary_target) + '_model_node_' + str(iter)].fit(X_train_filtered, y_train_filtered['binary_target'])

        # predict the target
        y_pred = model_dict[str(binary_target) + '_model_node_' + str(iter)].predict(X_val)
        
        print(f"{iter}/ {model_count} Iteration")
        # keep track of the node
        iter += 1

        # calculate the accuracy
        # accuracy = np.mean(y_pred == y_val)

        # print(f"Accuracy: {accuracy}")

    return model_dict

# we use downsampling here to balance the data for a minority class 
def balanced_bagging(X, y , target_column, rel_size_bagg_min, minority_class, num_bags = 3):
    """
    The funciton takes a df and returns n number of balanced bagging dataframes with the size of
    rel_size_bagg * minority_sample. It expects a binary target column 

    Args:
        X: dataframe with the features
        y: target
        rel_size_bagg_min: relative size of the minority class
        minority_class: the minority class
        num_bags: number of bags

    Returns:
        df_list: list of dataframes with each dataframe containing the minority and majority class in a balanced way
    """

    df_list = []
    # get the minority class
    minority_df = X[y[target_column] == minority_class].reset_index()
    majority_df = X[y[target_column] != minority_class].reset_index()


    # get the size of the minority class
    minority_size = minority_df.shape[0]


    for n in range(num_bags):
        # get a random sample of the majority class

        if rel_size_bagg_min == 1:
            minority_sample = minority_df
        else:
            # minority_class = minority_df.sample(frac = rel_size_bagg_min, )
            minority_sample = resample(minority_df, replace = False,  n_samples = int(minority_size * rel_size_bagg_min))

        majority_sample = resample(majority_df, replace = False,  n_samples = int(minority_size * rel_size_bagg_min))


        # combine the minority and majority sample
        df = pd.concat([minority_sample.set_index('Claim Identifier'), majority_sample.set_index('Claim Identifier')])

        # shuffle the df

        y_sample = y.loc[df.index]


        df_list.append((df, y_sample))

    return df_list

def model_evaluation(model, X_val, y_val):
    """
    This function takes in a model and returns the accuracy, precision, recall, f1 score and classification report
    
    Args:
        model: A model
        X_val: The validation set
        y_val: The validation set labels
        
    Returns:
        accuracy: The accuracy of the model
        precision: The precision of the model
        recall: The recall of the model
        f1: The f1 score of the model
        classification_report: The classification report of the model
    """
    
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    
    return accuracy, precision, recall, f1, report



# This model returns us predicitons for our N-stage model

def n_stage_pred(split_model_dict, split_list, X_input: pd.DataFrame):
    """

    Predicts using a dictionary of models for a multi-stage prediction pipeline. 

    Each model corresponds to a specific split defined in the `split_list`. At each stage, the function uses the 
    corresponding model to make predictions, and based on the prediction results, filters the dataset for subsequent stages.

    Args:
        model_dict (dict): Dictionary of models, where keys correspond to split names and values are the trained models.
        split_list (list): List of split names corresponding to target values (e.g., '2 Non Comp').
        X_input (pd.DataFrame): Input feature dataframe for predictions.
        
        # y_input (pd.DataFrame): True target values (optional, for future use).

    Returns:
        pd.DataFrame: A DataFrame containing predictions for each input row.

    """

    iteration_df = X_input.copy()

    # create a dataframe with one column that contains a string. The dataframe is the same length as the input dataframe and contains the same index
    y_pred = pd.DataFrame(index = X_input.index)
    # new column with prediction placeholdr. This will be replaced with the actual prediction for each index value 
    y_pred['N_stage_pred'] = 'pred_placeholder'
    

    # binary mode split is a bity confusing but it contains actually values like '2 Non Comp' which also is used as key for the dict.
    for binary_model_split in split_list:

        if binary_model_split not in split_model_dict:
            raise ValueError(f"Model for split '{binary_model_split}' not found in split_model_dict.")

        # can replace this with a function that takes treshold into account. E.g. XGBoost prob_pred
        pred = split_model_dict[binary_model_split].predict(iteration_df)

        # check where prediction is 1 
        pred_index_one = iteration_df[pred == 1 ].index
        # check where prediction is 0
        pred_index_zero = iteration_df[pred == 0 ].index

        # set values to the String e.g. "2. Non Comp" where the current model predicted 1
        y_pred.loc[pred_index_one, 'N_stage_pred'] = binary_model_split


        # Safe the index of the iteration_df
        # index = X_input.iteration_df

        # New dataframe for next model prediction where predictions are 0 
        iteration_df = iteration_df.loc[pred_index_zero].copy()


    return y_pred


def n_stage_pred_rest(split_model_dict, split_list, X_input: pd.DataFrame, rest_model, label_dict):

    y_pred = n_stage_pred(split_model_dict, split_list, X_input)

    # check where model has not predicted yet 
    pred_index_rest = y_pred[y_pred['N_stage_pred'] == 'pred_placeholder'].index

    iteration_df = X_input.loc[pred_index_rest].copy()

    # predict the rest of the values with the rest model
    pred = rest_model.predict(iteration_df)

    # define as pd.series
    pred = pd.Series(pred, index = iteration_df.index)

    # translate the predictions with the label_dict
    pred = pred.map(label_dict)


    # set the values to the last predictions 
    y_pred.loc[pred_index_rest, 'N_stage_pred'] = pred 


    return y_pred
        
