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

from imblearn.combine import SMOTETomek 

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
        false_negatives = true_values['Claim Injury Type'][(true_values[minority_class_col] == 1) & (prediction == 0)]

        # labels 
        labels = true_values['Claim Injury Type'].unique()
        # divide the value counts 
        true_values_negative = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0)].value_counts()
        true_values_positive = true_values['Claim Injury Type'][(true_values[minority_class_col] == 1)].value_counts()
        
        # count values for every  for the false positives and true negatives
        value_count_fp = false_positives.value_counts().astype(float)
        value_count_tn = true_negative.value_counts().astype(float)
        value_count_fn = false_negatives.value_counts().astype(float)


        # if a value is missing in the false positives or true negatives add it with a value of 0
        for value in labels:
            if value not in value_count_fp.index:
                value_count_fp[value] = 0
            if value not in value_count_tn.index:
                value_count_tn[value] = 0
            if value not in value_count_fn.index:
                value_count_fn[value] = 0


        # divide the value counts by the total value counts
        for value in true_values_negative.index:
            value_count_fp[value] = value_count_fp[value] / true_values_negative[value]
            value_count_tn[value] = value_count_tn[value] / true_values_negative[value]
        
        for value in true_values_positive.index:
            value_count_fn[value] = value_count_fn[value] / true_values_positive[value]
        

    else:
        false_positives = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0) & (prediction == 1)] 
        true_negative = true_values['Claim Injury Type'][(true_values[minority_class_col] == 0) & (prediction == 0)] 
        false_negative = true_values['Claim Injury Type'][(true_values[minority_class_col] == 1) & (prediction == 0)] 
        value_count_fp = false_positives.value_counts()
        value_count_tn = true_negative.value_counts()
        value_count_fn = false_negative.value_counts()


    return value_count_fp, value_count_tn, value_count_fn

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

def filter_target(y_train, binary_target, rest_binary_target):
        """
        Define a function that takes a dataframe and returns a dataframe with the rows that contain the binary_target and rest_binary_target values
        and a column that defines the binary_target values with 0 and 1

        Args:
            y_train: target dataframe
            binary_target: list of binary target values e.g. ['2 Non Comp']
            rest_binary_target: list of rest binary target values e.g. ['3. MED ONLY' , '4. TEMPORARY'] 

        Returns:
            y_train_filtered: dataframe with the rows that contain the binary_target and rest_binary_target values
            and a column that defines the binary_target values with 0 and 1
        
        """


        # filter the features | get all the rows where y_val contains the rest_binary_target values
        y_train_filtered = y_train[y_train['Claim Injury Type'].isin(rest_binary_target + binary_target)].copy()

        y_train_filtered.loc[:, 'binary_target'] = binary_splitting(y_train_filtered, binary_target, 'Claim Injury Type')

        return y_train_filtered


def SMOTE_sample_return(X, y, target_class: str):

    top_features = get_top_features_rf(X, y, target_class, n_features = 5)

    X_resampled, y_resampled = SMOTETomek(sampling_strategy=0.05, random_state=42).fit_resample(X[top_features], y[target_class])

    return X_resampled, y_resampled

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
        model_dict[binary_target[0]] = model_classes[model](**kwargs)


        # this gives back a dataframe with rows only containing values of the binary_target and rest_binary_target
        # and a column that defines the binary_target values with 0 and 1
        y_train_filtered = filter_target(y_train, binary_target, rest_binary_target)

        X_train_filtered = X_train.loc[y_train_filtered.index]
        print(y_train_filtered.head())  # Preview the data

        if (sum(y_train_filtered['binary_target'])/y_train_filtered.shape[0]) < 0.05:
        
        # get the resampled data with SMOTE and return it | Includes featrue selection with top features of random forest
            X_oversampled ,y_oversampled = SMOTE_sample_return(X_train_filtered, y_train_filtered, 'binary_target')

            print(y_oversampled)
        else:
            
            # get top features out of random forest classifier
            top_features = get_top_features_rf(X_train_filtered, y_train_filtered, 'binary_target')
            X_oversampled = X_train_filtered[top_features]
            y_oversampled = y_train_filtered['binary_target']   
        
        

        # train the model of the node with the filtered features and target values inside the dictionary
        model_dict[binary_target[0]].fit(X_oversampled, y_oversampled)

        # predict the target
        # y_pred = model_dict[binary_target[0]].predict(X_val)
        
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
def n_stage_pred(split_model_dict, split_list, X_input: pd.DataFrame, treshold_predict = False, treshold_value = [0.5]):
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
    
    # iterate through the treshold_value list
    treshold_index = 0
    # binary mode split is a bity confusing but it contains actually values like '2 Non Comp' which also is used as key for the dict.
    for binary_model_split in split_list:

        if binary_model_split not in split_model_dict:
            raise ValueError(f"Model for split '{binary_model_split}' not found in split_model_dict.")
        
        if treshold_predict:
            
            if len(treshold_value) == 1:

                # get columns that have been used for training 
                train_feat = split_model_dict[binary_model_split].feature_names_in_
                # predict with the treshold value 
                pred = split_model_dict[binary_model_split].predict_proba(iteration_df[train_feat])[:, 1]

                # Turn the boolean prediction into a binary value
                pred = (pred > treshold_value[0]).astype(int)
            else:
                
                # predict with each value for the treshold
                train_feat = split_model_dict[binary_model_split].feature_names_in_
                pred = split_model_dict[binary_model_split].predict_proba(iteration_df[train_feat])[:, 1]
                pred = (pred > treshold_value[treshold_index]).astype(int)

        else:
        # can replace this with a function that takes treshold into account. E.g. XGBoost prob_pred
            
            train_feat = split_model_dict[binary_model_split].feature_names_in_
            pred = split_model_dict[binary_model_split].predict(iteration_df)[train_feat]

        # increase the treshold index to get the next treshold value in next iteration
        treshold_index += 1
        
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



def get_top_features_rf(X_train, y_train,  
                        target_class,
                        n_features = 10,
                        sample_fraction=None,   
                        n_estimators=100, 
                        random_state=42):
    """
    Get top n important features using Random Forest Classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.DataFrame
        Training labels
    n_features : int
        Number of top features to return
    target_class : str
        Target class to predict
    sample_fraction : float, optional (default = None)
        values ranging (0,1). Uses a sample to create feat importance        
    n_estimators : int, optional (default=100)
        Number of trees in the forest
    random_state : int, optional (default=42)
        Random state for reproducibility
        
    Returns:
    --------
    np.array
        Array of top n feature names
    """
    if sample_fraction:
    # Get indices for each class
        pos_idx = y_train[y_train[target_class] == 1].index
        neg_idx = y_train[y_train[target_class] == 0].index
        
        # Calculate samples needed for each class
        n_pos = int(len(pos_idx) * sample_fraction)
        n_neg = int(len(neg_idx) * sample_fraction)
        
        # Sample from each class
        pos_sample = np.random.choice(pos_idx, size=n_pos, replace=False)
        neg_sample = np.random.choice(neg_idx, size=n_neg, replace=False)
        
        # Combine samples
        sample_idx = np.concatenate([pos_sample, neg_sample])
        X_train_sample = X_train.iloc[sample_idx]
        y_train_sample = y_train.iloc[sample_idx]
    else:
        X_train_sample = X_train
        y_train_sample = y_train



    # Initialize and train Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                              random_state=random_state)
    rf.fit(X_train_sample, y_train_sample[target_class])
    
    # Evaluate model
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    })
    
    # Rank and select top features
    feature_importance_df = feature_importance_df.sort_values(
        by='Importance', ascending=False)
    
    return feature_importance_df['Feature'][:n_features].values



def n_stage_pred_rest(split_model_dict, split_list, X_input: pd.DataFrame, rest_model, label_dict, treshold_predict = False, treshold_value = [0.5]):

    y_pred = n_stage_pred(split_model_dict, split_list, X_input, treshold_predict = treshold_predict, treshold_value = treshold_value)

    # check where model has not predicted yet 
    pred_index_rest = y_pred[y_pred['N_stage_pred'] == 'pred_placeholder'].index

    # get the columns that have been used for training
    train_feat = rest_model.feature_names_in_
    # get the values that have not been predicted yet
    iteration_df = X_input.loc[pred_index_rest].copy()

    # predict the rest of the values with the rest model
    pred = rest_model.predict(iteration_df[train_feat])

    # define as pd.series
    pred = pd.Series(pred, index = iteration_df.index)

    # translate the predictions with the label_dict
    pred = pred.map(label_dict)


    # set the values to the last predictions 
    y_pred.loc[pred_index_rest, 'N_stage_pred'] = pred 


    return y_pred


# this functions shows in each step how many of the target values we keep and how many we lose
# def value_loss_keep()
    # for target_split in model_split_list:
    #     print(f"for {target_split[0][0]} we are losing/winning the folliwng values in this iteration")

    # (model_dict['2. NON-COMP'].predict_proba(X_val_encoded)[:,1] > 1).astype(int)

    # min_class_list = ['2. NON-COMP']
    # y_val['minority_class_node_1'] = binary_splitting(y_val, min_class_list, 'Claim Injury Type')
    # # get predictions for the first node
    # predictions_node_1 = model_dict['2. NON-COMP'].predict(X_val_encoded)
    # # get confusion_matrix
    # print('no treshold given')
    # print(confusion_matrix(y_val['minority_class_node_1'].values, predictions_node_1))

    # predictions_node_1_tresh =  (model_dict['2. NON-COMP'].predict_proba(X_val_encoded)[:,1] > 0.85).astype(int)

    # print('treshold given')
    # print(confusion_matrix(y_val['minority_class_node_1'].values, predictions_node_1_tresh))

    # plot = sns.heatmap(confusion_matrix(y_val['minority_class_node_1'].values, predictions_node_1_tresh), annot=True, fmt='d', cmap='Blues')



# use the filter_target
# 
def information_loss_n_stage(model_dict, model_split_list, X_val_encoded, y_val, treshold_predict = False, treshold_value = [0.5]):
    # increase the treshold index to get the next treshold value in next iteration
    treshold_index = 0 
    for target_split in model_split_list:
        print(f'We are prediciton for either {target_split[0]} = 1 \n and for: \n {target_split[1]} = 0')
        print(f"for this split we are losing/keeping the followng values in this iteration\n")
        

    #     we filter for values that are in the target of target_split[0] and target_split[1]
        y_iteration_target = filter_target(y_val, target_split[0], target_split[1])

        X_iteration_values = X_val_encoded.loc[y_iteration_target.index]


        if treshold_predict:
        
            if len(treshold_value) == 1:
                iteration_pred = model_dict[target_split[0][0]].predict_proba(X_iteration_values)[:,1] > treshold_value[0]
                iteration_pred = iteration_pred.astype(int)
            
            else:
            
                iteration_pred = model_dict[target_split[0][0]].predict_proba(X_iteration_values)[:,1] > treshold_value[0]
                iteration_pred = iteration_pred.astype(int)
                # increase the treshold index to get the next treshold value in next iteration
                treshold_index += 1

        else:
        # get the predictions for the medical model
            iteration_pred = model_dict[target_split[0][0]].predict(X_iteration_values)


    # get report for the medical model
        print(classification_report(y_iteration_target['binary_target'], iteration_pred))

        fp, tn, fn = fp_tn_distribution(iteration_pred, y_iteration_target, minority_class_col = 'binary_target', normalize = True)

    # fp are values which we lose
        print('we are loosing the following values in this step:' , fp)
        

        print('we are keeping the following values in this step:' , tn, '\n')

        print('the following values are misclassified as Negatives:', fn)
        # tn are values which we keep`

    return None