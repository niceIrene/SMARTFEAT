# %% data loading
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import pathlib as Path
from SMARTFEAT.serialize import *
from SMARTFEAT.feature_evaluation import feature_evaluation_show_all
import pandas as pd 
import numpy as np
from SMARTFEAT.Prediction_helper import *
from sklearn.model_selection import train_test_split
from SMARTFEAT.search import *
import copy
from SMARTFEAT.feature_evaluation import *
data_df = pd.read_csv("../dataset/[DatasetPath]/[DatasetWithNewFeatures].csv")
y_label = 'Y_Label'
# %% data general preprocessing
data_df, features = data_preproessing(data_df, y_label)
X = data_df[features]
y = data_df[y_label]
# %% test helpful information
print("===========================================")
print('mutual info')
feature_evaluation_show_all(X, y, 'mutual info')
print("===========================================")
print('rfe-rf')
feature_evaluation_show_all(X, y, 'rfe-rf')
print("===========================================")    
print('feature_importance')
feature_evaluation_show_all(X, y, 'feature_importance')
print("===========================================")    
# %% Obtain prediction outcome
X_train, X_test, y_train, y_test =train_test_split(data_df[features],data_df[y_label],
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=data_df[y_label])
models = GetBasedModel()
print(len(X_train))
names,results, tests = PredictionML(X_train, y_train,X_test, y_test,models)
basedLineScore = ScoreDataFrame(names,results, tests)
basedLineScore

# %% reconstruction of groupby features trainset
for c in data_df.columns:
    if 'GROUPBY' in c and y_label in c:
        data_df[c]= 0

def obtain_groupby_cols(c):
    print(c)
    pattern = r'GROUPBY_\[(.*?)\]_(.*?)_(.*)'
    match = re.match(pattern, c)
    if not match:
        raise ValueError("Invalid input format")
    groupby_col = [col.strip(" '[]") for col in match.group(1).split(',')]
    function = match.group(2)
    agg_col = match.group(3)
    return groupby_col, agg_col, function    

# recompute the groupby information from the train set
train_df = X_train.join(y_train)
for c in X_train.columns:
    if 'GROUPBY' in c and y_label in c:
        print(c)
        # obtain the groupby columns from c
        groupby_col, agg_col, function = obtain_groupby_cols(c)
        X_train[c] = train_df.groupby(groupby_col)[agg_col].transform(function)
        # heuristically remove the attributes c if it is hard to perform imputation 
        group_num = len(X_train[groupby_col].drop_duplicates())
        print(group_num)
        if group_num > 0.5 * len(X_train):
            print("Drop the attribute as it is impossible to impute")
            X_train = X_train.drop([c], axis = 1)
            X_test = X_test.drop([c], axis = 1)
# %% impute the test set from the trainset groupby information
groupby_features = []
for c in X_test.columns:
    if 'GROUPBY' in c and y_label in c:
        X_test = X_test.drop([c],axis = 1)
        groupby_features.append(c)
print("All features containing groupby are")
print(groupby_features)
# dropped the hard to impute columns
for c in groupby_features:
    groupby_col, agg_col, function = obtain_groupby_cols(c)
    # Create a dictionary to store the default values for each unique combination of categories
    aggregations = train_df.groupby(groupby_col)[agg_col].agg(function).reset_index()
    # just to make sure the merge is successful
    X_test[y_label] = 0
    # Impute missing values in the test set using the train set aggregations or default values
    X_test = X_test.merge(aggregations, on=groupby_col, how='left', suffixes=('', '_impute')) 
    X_test = X_test.rename(columns={y_label + '_impute': c})
    # if cannot find such a match, we will obtain a nan value in a column 
    if X_test[c].isna().any():
        nan_rows_count = X_test[X_test[c].isna()].shape[0]
        print("-----------")
        print(c)
        print("Cannot find a match in the train set")
        print("NAN rows count: ", nan_rows_count)
        # if more than half of the test set needs to impute, drop the column, otherwise use the aggregate value to impute it)
            # we use the aggregate information of the entire dataset for imputation
        X_test[c].fillna(train_df[agg_col].agg(function), inplace=True)
if len(groupby_features) > 0:
    X_test = X_test.drop([y_label],axis = 1)
X_test
# %% final accuracy with the dataset imputation
X_test = X_test[list(X_train.columns)]
models = GetBasedModel()
print(len(X_train))
names,results, tests = PredictionML(X_train, y_train,X_test, y_test,models)
# %%