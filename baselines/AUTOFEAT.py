# %% test helpful information
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from SMARTFEAT.serialize import *
from SMARTFEAT.feature_evaluation import feature_evaluation_show_all
import pandas as pd 
import numpy as np
from SMARTFEAT.Prediction_helper import *
from sklearn.model_selection import train_test_split
from SMARTFEAT.search import *
from SMARTFEAT.feature_evaluation import *
from autofeat import AutoFeatClassifier

data_df = pd.read_csv("../dataset/[DatasetPath]/[DatasetWithNewFeatures].csv")
y_label = 'Y_Label'
# %% data general preprocessing
data_df, features = data_preproessing(data_df, y_label)

X = data_df[features]
y = data_df[y_label]
# %% splitting training and testing set
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
# %% Obtain new prediction outcome

models = GetBasedModel()
print(len(X_train_tr))
names,results, tests = PredictionML(X_train_tr, y_train,X_test_tr, y_test,models)
basedLineScore = ScoreDataFrame(names,results, tests)
basedLineScore
# %%
