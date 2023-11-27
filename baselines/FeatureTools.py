# %% test helpful information
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from serialize import *
from feature_evaluation import feature_evaluation_show_all, feature_evaluation_select_k
import pandas as pd 
import numpy as np
from Prediction_helper import *
from sklearn.model_selection import train_test_split
from search import *
from feature_evaluation import *
import featuretools as ft

data_df = pd.read_csv("../dataset/[DatasetPath]/[DatasetWithNewFeatures].csv")
y_label = 'YLabel'
attributes = list(data_df.columns)
attributes.remove(y_label)
features = attributes
X = data_df[features]
y = data_df[y_label]
# %%
es = ft.EntitySet(id="data")
es = es.add_dataframe(
    dataframe_name="data",
    dataframe=X,
    index="index",
)
features_df, feature_dfs = ft.dfs(entityset=es,
                               target_dataframe_name='data',
                               agg_primitives=['count', 'sum', 'mean', 'max'],
                               trans_primitives=['add_numeric', 'multiply_numeric'],
                               verbose=True, max_depth=2)
features_df
feature_dfs
# %%
X = features_df
data_df = pd.concat([X, y], axis=1)
data_df
# %%
for c in data_df.columns:
    if type(data_df[c][0]) != np.int64 and type(data_df[c][0]) != np.float64:
        print(type(data_df[c][0]))
        data_df[c] = data_df[c].astype(object)
        data_df[c], _  = pd.factorize(data_df[c])
        # factorize these columns
data_df = data_df.replace([np.inf, -np.inf], np.nan)
data_df = data_df.dropna()

for c in data_df.columns:
    if type(data_df[c][0]) != np.int64 and type(data_df[c][0]) != np.float64:
        print(type(data_df[c][0]))
# %% 
for index, row in X.iterrows():
    for column in X.columns:
        if row[column] < 0:
            print(f"Cell at ({index}, {column}): {row[column]}")

print("===========================================")
print('mutual info')
feature_evaluation_show_all(X, y, 'mutual info')

print("===========================================")
print("chi2")
feature_evaluation_show_all(X, y, 'chi2')

print("===========================================")
print('rfe-rf')
feature_evaluation_show_all(X, y, 'rfe-rf')
print("===========================================")    
print('feature_importance')
feature_evaluation_show_all(X, y, 'feature_importance')
print("===========================================")    

# %% Obtain new prediction outcome
attributes = list(data_df.columns)
attributes.remove('YLabel')
features = attributes
X_train, X_test, y_train, y_test =train_test_split(data_df[features],data_df['YLabel'],
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=data_df['YLabel'])
models = GetBasedModel()
print(len(X_train))
names,results, tests = PredictionML(X_train, y_train,X_test, y_test,models)
basedLineScore = ScoreDataFrame(names,results, tests)
basedLineScore
# %% feature selection
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)

ft.selection.remove_highly_null_features(data_df)
new_data_df, new_features = remove_single_value_features(data_df, features=feature_dfs)
new_data_df, new_features = remove_highly_correlated_features(new_data_df, features=feature_dfs)
new_data_df
# %%
attributes = list(new_data_df.columns)
attributes.remove(y_label)
features = attributes
X_train, X_test, y_train, y_test =train_test_split(new_data_df[features],new_data_df[y_label],
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=new_data_df[y_label])
models = GetBasedModel()
print(len(X_train))
names,results, tests = PredictionML(X_train, y_train,X_test, y_test,models)
basedLineScore = ScoreDataFrame(names,results, tests)
basedLineScore