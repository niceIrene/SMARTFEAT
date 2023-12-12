import pandas as pd
import numpy as np
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import copy

def clean_data(data_df):
    data_temp = copy.copy(data_df)
    for c in data_temp.columns:
        if type(data_temp[c][0]) != np.int64 and type(data_temp[c][0]) != np.float64:
            print(type(data_temp[c][0]))
            data_temp[c] = data_temp[c].astype(object)
            data_temp[c], _  = pd.factorize(data_temp[c])
            # factorize these columns
    data_temp = data_temp.replace([np.inf, -np.inf], np.nan)
    data_temp = data_temp.dropna()

    for c in data_temp.columns:
        if type(data_temp[c][0]) != np.int64 and type(data_temp[c][0]) != np.float64:
            print(type(data_temp[c][0]))
    return data_temp


def split_train_test(data,test_ratio):
    np.random.seed(0)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def get_train_test(data, split, list_cols, y_label):
  train_set,test_set = split_train_test(data,split)
#   print(len(train_set), "train +", len(test_set), "test")
  train_x = pd.DataFrame(train_set, columns = list_cols)
  train_label = train_set[y_label]
  test_x = pd.DataFrame(test_set, columns = list_cols)
  test_label = test_set[y_label]
  return train_x, test_x, train_label, test_label, train_set, test_set

def GetBasedModel():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('NB'   , GaussianNB()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    basedModels.append(('ET'   , ExtraTreesClassifier()))
    basedModels.append(('DNN', MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.0001, alpha=0.001, max_iter=1000)))
    return basedModels

def PredictionML(X_train, y_train, X_test, y_test, models):
    num_folds = 10
    scoring = 'roc_auc'
    results = []
    names = []
    tests = []
    for name, model in models:
        # uncomment this part if the training for large dataset takes a long time
        kfold = StratifiedKFold(n_splits=num_folds, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        # compute the test score
        model.fit(X_train, y_train)
        test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        # print only the result for the test score
        # msg = "%s: %f (%f) - test score %f" % (name, , , test_score)
        # print everything
        msg = "%s: %f (%f) - test score %f" % (name, cv_results.mean(),cv_results.std(), test_score)
        print(msg)
        tests.append(test_score)      
    return names, results, tests

def ScoreDataFrame(names,results, tests):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
        return float(prc.format(f_val))
    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))
    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores, 'TestResult':tests})
    return scoreDataFrame