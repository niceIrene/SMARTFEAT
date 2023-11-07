#  include packages
# CML
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
sys.path.append("./")
sys.path.append('../prompts/')
import pathlib as Path
from serialize import *
import os
import argparse
import openai
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory
from search import CurrentAttrLst
import pandas as pd 
import numpy as np
from Prediction_helper import *
from sklearn.model_selection import train_test_split
from search import *
import copy
from feature_evaluation import *
import time


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type= str, default='./dataset/pima_diabetes/')
    args.add_argument('--predict_col', type=str, default='Outcome')
    args.add_argument('--csv', type=str, default='diabetes.csv')
    args.add_argument('--model', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003'], default='gpt-3.5-turbo')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--n_generate_sample', type=int, default=1)
    args.add_argument('--clf_model', type=str, default='Decision Tree')
    # 100 means does not select any feature, by default use mutual information to select feature
    args.add_argument('--feature_selection', type=int, default=100)
    args.add_argument('--delimiter', type=int, default=1)
    args.add_argument('--sampling_budget', type=int, default=10)
    args = args.parse_args()
    return args

args = parse_args()

if args.delimiter ==1:
    data_df = pd.read_csv(args.path + args.csv, delimiter=',')
else:
    data_df = pd.read_csv(args.path + args.csv, delimiter=';')

attributes = list(data_df.columns)
attributes.remove(args.predict_col)
org_features = attributes
with open(args.path+ 'data_agenda.txt', 'r') as f:
    data_agenda = f.read()
f.close

print("The original features are")
print(org_features)

# print("The original predication results are")
# data_prelim =clean_data(data_df)
# X_train, X_test, y_train, y_test =train_test_split(data_prelim[org_features],data_prelim[args.predict_col],
#                                                    test_size=0.25,
#                                                    random_state=0,
#                                                    stratify=data_prelim[args.predict_col])
# models = GetBasedModel()
# names,results, tests = PredictionML(X_train, y_train,X_test, y_test,models)

# initialize the root state 
cur_attr_lst = CurrentAttrLst(org_features, data_agenda, data_df, args.clf_model, args.sampling_budget)
print("The current step is!!!!!!")
print(cur_attr_lst.step)
while True:
    try:
        if cur_attr_lst.step < len(org_features):
            result_lst = -1
            result_lst = feature_generator_propose(cur_attr_lst, org_features, args.predict_col)
            if result_lst == -1:
                cur_attr_lst.step += 1
                cur_attr_lst.last_op = None
                continue
            if result_lst is None:
                continue
            for r in result_lst:
                print("Start value evaluation")
                print(r)
                state_evaluator(r, cur_attr_lst, args.predict_col)
        else:
            result_lst = feature_genetor_cot(cur_attr_lst, args.predict_col, args.temperature, args.n_generate_sample)
            if result_lst == -1 or cur_attr_lst.budget_cur >= cur_attr_lst.budget:
                cur_attr_lst.budget_cur = 0
                # more than three continuous failures or reach budget
                if isinstance(cur_attr_lst.last_op, MultiExtractor):
                    print("Search process ends")
                    break
                elif isinstance(cur_attr_lst.last_op, BinaryOperatorAlter):
                    # for binary operator reaches the generation error times.
                    print("Binary ends, go to aggregator")
                    cur_attr_lst.previous_two = 0
                    cur_attr_lst.last_op = AggregateOperator(cur_attr_lst.data_agenda, cur_attr_lst.model, args.predict_col, cur_attr_lst.cur_attr_lst)
                    continue
                elif isinstance(cur_attr_lst.last_op, AggregateOperator):
                    # for binary operator reaches the generation error times.
                    print("Aggregate ends, go to extract")
                    cur_attr_lst.previous_two = 0
                    cur_attr_lst.last_op = MultiExtractor(cur_attr_lst.data_agenda, cur_attr_lst.model, args.predict_col, cur_attr_lst.cur_attr_lst)
                    continue
            elif result_lst is None or len(result_lst) == 0:
                print("result lst is empty")
                cur_attr_lst.previous_two += 1
                continue
            else:
                for r in result_lst:
                    state_evaluator(r, cur_attr_lst, args.predict_col)
    except Exception as e:
        print("!!!!!")
        print(e)
        wait_time = 2  # Delay in seconds
        time.sleep(wait_time)
        continue