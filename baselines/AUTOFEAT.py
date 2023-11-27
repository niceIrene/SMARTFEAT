# %% test helpful information
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from serialize import *
from feature_evaluation import feature_evaluation_show_all
import pandas as pd 
import numpy as np
from Prediction_helper import *
from sklearn.model_selection import train_test_split
from search import *
from feature_evaluation import *
from autofeat import AutoFeatClassifier


data_df = pd.read_csv("../dataset/[DatasetPath]/[DatasetWithNewFeatures].csv")
y_label = 'Y_Label'

attributes = list(data_df.columns)
attributes.remove(y_label)
features = attributes
X = data_df[features]
y = data_df[y_label]
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
# splitting training and testing set
X_train, X_test, y_train, y_test =train_test_split(data_df[features],data_df[y_label],
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=data_df[y_label])
afreg = AutoFeatClassifier(verbose=1, feateng_steps=2)
X_train_tr = afreg.fit_transform(X_train, y_train)
X_test_tr = afreg.transform(X_test)
print("autofeat new features:", len(afreg.new_feat_cols_))
# %%
X = pd.concat([X_train_tr, X_test_tr], axis= 0)
print(X)

# %% find all negative values.
for index, row in X.iterrows():
    for column in X.columns:
        if row[column] < 0:
            print(f"Cell at ({index}, {column}): {row[column]}")

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
# %% Obtain new prediction outcome

models = GetBasedModel()
print(len(X_train_tr))
names,results, tests = PredictionML(X_train_tr, y_train,X_test_tr, y_test,models)
basedLineScore = ScoreDataFrame(names,results, tests)
basedLineScore