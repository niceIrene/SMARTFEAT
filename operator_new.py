import os
import sys
sys.path.append('../')
sys.path.append("./")
sys.path.append('../prompts/')
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from gpt import *
from prompt import *
import json
import re
from serialize import *
import pandas as pd

class Operator(object):
    def __init__(self, agenda,model_prompt, y_attr):
        self.data_agenda = agenda
        self.y_attr = y_attr
        self.model_prompt = model_prompt

class UnaryOperator(Operator):
    def __init__(self, agenda, model_prompt, y_attr, org_attr):
        super().__init__(agenda, model_prompt, y_attr)
        self.org_attr = org_attr
    def __str__(self):
        return f"Unary operator, original feature '{self.org_attr}' to predict {self.y_attr}."
    def generate_new_feature(self, temp):
        rel_agenda = obtain_rel_agenda([self.org_attr], self.data_agenda)
        y_agenda = obtain_rel_agenda([self.y_attr], self.data_agenda)
        data_prompt = rel_agenda + ', ' + y_agenda + '\n'
        op_prompt = unary_prompt_propose.format(y_attr= self.y_attr, input = self.org_attr)
        prompt = "Attribute description: " + data_prompt + "Downstream machine learning models: " + self.model_prompt +'\n'+ op_prompt
        # for propose, generate one candidate. in the answer, it gives a list of proposals.
        res = gpt_fix_or_propose(prompt = prompt, n = 1, temperature=temp)
        # updated to return a list of the results.
        res_lst = []
        for r in res:
            if 'certain' in r or 'high ' in r:
                result_dict = {}
                result_dict['new_feature'] = "Unary_{}".format(self.org_attr)
                result_dict['description'] = r
                result_dict['relevant'] = self.org_attr
                res_lst.append(result_dict)
        if len(res_lst)==0:
            res_lst = None
        print(res_lst)
        return res_lst
    def find_function(self, result_dict, temp=0.1):
        # handle one-hot-encoding as specific case:
        if 'encoding' in result_dict['description'] or 'Encoding' in result_dict['description']:
            func_str = 'encoding'
            return func_str
        new_col, rel_cols, descr, rel_agenda = self.parse_output(result_dict)
        data_prompt = rel_agenda
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["data_prompt", "new_feature", "relevant", "description"],
        template="You are a data scientist specializing in feature engineering, where you excel in finding the most suitable operation to obtain new features based on attribute context.\
        Attribute description: {data_prompt}, generate the most appropriate python function to obtain new feature(s) {new_feature} (output) using feature {relevant} (input), function description: {description}.")
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"data_prompt": data_prompt, "new_feature": new_col, "relevant": rel_cols, "description": descr})
        print(func_str)
        start = func_str.find('def')
        func_str = func_str[start:]
        return func_str
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = obtain_relevant_cols(result_dict)
        rel_agenda = obtain_rel_agenda(rel_cols,self.data_agenda)
        return new_col, descr, rel_cols, rel_agenda        
        
class Descrtizer(UnaryOperator):
    def __str__(self):
        return f"Bucketize '{self.org_attr}' to predict '{self.y_attr}"
    def generate_new_feature(self, temp):
        # make sure all attributes here is appropriate for descrize
        rel_agenda = obtain_rel_agenda([self.org_attr], self.data_agenda)
        y_agenda = obtain_rel_agenda([self.y_attr], self.data_agenda)
        data_prompt = rel_agenda + ', ' + y_agenda + '\n'
        op_prompt = descritizer_prompt_fix.format(y_attr= self.y_attr, input = self.org_attr)
        prompt = "Attribute description: " + data_prompt + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        prompt = data_prompt + op_prompt
        answer = gpt_fix_or_propose(prompt = prompt, n = 1, temperature=temp)
        if 'Yes' in answer[0]:
            result_dict = {}
            result_dict['new_feature'] = "Bucketized_{}".format(self.org_attr)
            result_dict['description'] = "Bucketized {}".format(self.org_attr)
            result_dict['relevant'] = self.org_attr
        else:
            result_dict = None
        return result_dict
    def find_function(self, result_dict, temp=0.1):
        new_col, rel_cols, descr, rel_agenda = self.parse_output(result_dict)
        data_prompt = rel_agenda
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["data_prompt", "new_feature", "relevant", "description"],
        template="You are a data scientist specializing in feature engineering, where you excel in finding the most suitable operation to obtain new features based on attribute context.\
            Attribute description: {data_prompt}, generate the most appropriate python function \
            to obtain new feature {new_feature} (output) using feature {relevant} (input), function description: {description}. Do not provide a lambda function."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"data_prompt": data_prompt, "new_feature": new_col, "relevant": rel_cols, "description": descr})
        start = func_str.find('def')
        func_str = func_str[start:]
        return func_str


class Normalizer(UnaryOperator):
    def __str__(self):
        return f"Normalizer operator, original feature '{self.org_attr}' to predict {self.y_attr}."
    def generate_new_feature(self, temp):
        # determine yes or no
        rel_agenda = obtain_rel_agenda([self.org_attr], self.data_agenda)
        y_agenda = obtain_rel_agenda([self.y_attr], self.data_agenda)
        data_prompt = rel_agenda + ', ' + y_agenda + '\n'
        op_prompt = normalizer_prompt_fix.format(y_attr= self.y_attr, input = self.org_attr)
        prompt = "Attribute description: " + data_prompt + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        answer = gpt_fix_or_propose(prompt = prompt, n = 1, temperature=temp)
        if 'Yes' in answer[0]:
            result_dict = {}
            result_dict['new_feature'] = "Normalized_{}".format(self.org_attr)
            result_dict['description'] = "Normalize {}".format(self.org_attr)
            result_dict['relevant'] = self.org_attr
        else:
            result_dict = None
        return result_dict
    def find_function(self, result_dict):
        # by default, we consider min-max normalization. Other options, standardazation, robust scaling, max absolute scaling.
        if result_dict is not None:
            func_str = '''def normalize_A(A):
                A_min = min(A)
                A_max = max(A)
                A_normalized = (A - A_min) / (A_max - A_min)
                return A_normalized'''
            return func_str
        else:
            print("Not suitable for normalize")
            return None
        
class RowExpand(UnaryOperator):
    def __str__(self):
        return f"Row Expand operator, original feature '{self.org_attr}' to predict {self.y_attr}."
    def generate_new_feature(self, temp):
        rel_agenda = obtain_rel_agenda([self.org_attr], self.data_agenda)
        y_agenda = obtain_rel_agenda([self.y_attr], self.data_agenda)
        data_prompt = rel_agenda + ',' + y_agenda + '\n'
        op_prompt = rowexpand_promp_fix.format(y_attr= self.y_attr, input = self.org_attr)
        prompt = "Attribute description: " + data_prompt + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        answer = gpt_fix_or_propose(prompt = prompt, n = 1, temperature=temp)        
        if 'Yes' in answer[0]:
            return True
        else:
            return None
    def find_function(self, temp = 0.1):
        rel_cols = [self.org_attr]
        rel_agenda = obtain_rel_agenda(rel_cols, self.data_agenda)
        data_prompt = rel_agenda
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["data_prompt", "relevant"],
        template="Attribute description: {data_prompt}, generate the most appropriate python function to split {relevant} into multiple columns."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"data_prompt": data_prompt, "relevant": rel_cols})
        start = func_str.find('def')
        func_str = func_str[start:]
        return func_str


class BinaryOperator(Operator):
    def __init__(self, agenda, model_prompt, y_attr, org_attrs):
        super().__init__(agenda, model_prompt, y_attr)
        self.org_attrs = org_attrs
    def __str__(self):
        return f"binary operator, using {str(self.org_attrs)}."
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = obtain_relevant_cols(result_dict)
        rel_agenda = obtain_rel_agenda(rel_cols,self.data_agenda)
        return new_col, descr, rel_cols, rel_agenda
    def generate_new_feature(self, temp):
        rel_agenda = obtain_rel_agenda(self.org_attrs, self.data_agenda)
        y_agenda = obtain_rel_agenda([self.y_attr], self.data_agenda)
        data_prompt = rel_agenda + ',' + y_agenda + '\n'
        op_prompt = binary_prompt_propose.format(y_attr= self.y_attr, input = self.org_attrs)
        prompt = "Attribute description: " + data_prompt + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        res = gpt_fix_or_propose_binary(prompt = prompt, n = 1, temperature=temp)
        res_lst = []
        for r in res:
            if 'certain' in r:
                result_dict = {}
                result_dict['new_feature'] = "Binary_{}".format(str(self.org_attrs))
                result_dict['description'] = "Binary operator {}".format(res[0])
                relevant_str = str(self.org_attrs[0]) +',' + str(self.org_attrs[1])
                result_dict['relevant'] = relevant_str
                res_lst.append(result_dict)
        if len(res_lst) == 0:
            res_lst = None
        return res_lst
    def find_function(self, result_dict, temp=0.1):
        new_col, rel_cols, descr, rel_agenda = self.parse_output(result_dict)
        data_prompt = rel_agenda
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["data_prompt", "new_feature", "relevant", "description"],
        template="You are a data scientist specializing in feature engineering, where you excel in finding the most suitable operation to obtain new features based on attribute context.\
            Attribute description: {data_prompt}, generate the most appropriate python function with +/-/*//to obtain new feature {new_feature} (output) using features {relevant} (input), function description: {description}. If the selected attribute is /, Handle the case of devide by zero."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"data_prompt": data_prompt, "new_feature": new_col, "relevant": rel_cols, "description": descr})
        start = func_str.find('def')
        func_str = func_str[start:]
        return func_str
    
class BinaryOperatorAlter(Operator):
    def __init__(self, agenda, model_prompt, y_attr):
        super().__init__(agenda, model_prompt, y_attr)
    def __str__(self):
        return f"binary operator."
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = result_dict['relevant']
        return new_col, descr, rel_cols
    def generate_new_feature(self, temp):
        op_prompt = binary_prompt_sampling.format(y_attr= self.y_attr)
        prompt = "Existing feature description: " + self.data_agenda + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt        
        answer = gpt_cot_binary(prompt = prompt, n = 1, temperature=temp)
        res_lst = []
        for i in range(len(answer)):
            # try:
            res_dic = re.search(r'\{[^}]+\}', answer[i])
            answer_dict = eval(res_dic.group(0))
            res_lst.append(answer_dict)
        # except:
            # print("result cannot parse")
            # res_lst.append(None)
            print("Result list is")
        return res_lst
        # print("The result list is")
        # print(res_lst)
        # return res_lst
    def find_function(self, result_dict, temp=0.1):
        print("Find function binary")
        new_col, descr, rel_cols = self.parse_output(result_dict)
        data_prompt = self.data_agenda
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["data_prompt", "new_feature", "relevant", "description"],
        template="You are a data scientist specializing in feature engineering, where you excel in finding the most suitable operation to obtain new features based on attribute context.\
            Attribute description: {data_prompt}, generate the most appropriate python function with +/-/*//to obtain new feature {new_feature} (output) using features {relevant} (input), function description: {description}. If the selected attribute is /, Handle the case of devide by zero."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"data_prompt": data_prompt, "new_feature": new_col, "relevant": rel_cols, "description": descr})
        start = func_str.find('def')
        func_str = func_str[start:]
        print(func_str)
        return func_str

class AggregateOperator(Operator):
    def __init__(self, agenda, model_prompt, y_attr, org_attrs):
        self.data_agenda = agenda
        self.y_attr = y_attr
        self.model_prompt = model_prompt
        self.response_schema = [
            ResponseSchema(name="groupby_col", description="The groupby columns, a list"),
            ResponseSchema(name="agg_col", description="The aggregate column"),
            ResponseSchema(name="function", description="the aggregation function")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()
    def __str__(self):
        # print prompt
        return f"groupby-aggregation operator to obtain a new feature to predict '{self.y_attr}' using {str(self.org_attrs)}."
    def generate_new_feature(self, temp, n):
        op_prompt = aggregator_prompt_cot.format(y_attr= self.y_attr)
        prompt = "Existing feature description: " + self.data_agenda + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        answer = gpt_cot_agg(prompt = prompt, temperature=temp)
        res_lst = []
        for i in range(len(answer)):
            try:
                res_dic = re.search(r'{(.*?)}', answer[i])
                answer_dict = eval(res_dic.group(0))
                answer_dict['new_feature'] = 'GROUPBY_' + str(answer_dict['groupby_col']) + '_' + str(answer_dict['function']) + '_' + str(answer_dict['agg_col'])
                print(answer_dict)
                res_lst.append(answer_dict)
            except:
                print("result cannot parse")
                res_lst.append(None)
        return res_lst


# class AggregateOperator(Operator):
#     def __init__(self, agenda, y_attr, org_attrs):
#         self.org_attrs = org_attrs
#         self.data_agenda = agenda
#         self.y_attr = y_attr
#         self.response_schema = [
#             ResponseSchema(name="groupby_col", description="The groupby columns, a list"),
#             ResponseSchema(name="agg_col", description="The aggregate column"),
#             ResponseSchema(name="function", description="the aggregation function")
#         ]
#         self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
#         self.format_instructions = self.output_parser.get_format_instructions()
#     def __str__(self):
#         # print prompt
#         return f"groupby-aggregation operator to obtain a new feature to predict '{self.y_attr}' using {str(self.org_attrs)}."
#     def generate_new_feature(self, temp, n):
#         res_lst = []
#         data_prompt = self.data_agenda
#         llm = OpenAI(temperature=temp, model_name = 'text-davinci-003', best_of=10, n=n)
#         prompt = PromptTemplate(
#             template="You are a data scientist specializing in feature engineering, where you excel in finding the most suitable groupby operation to obtain new features based on attribute context.\
#             Attribute description: {data_prompt}. Obtain a reasonable new feature by applying 'df.groupby(groupby_col)[agg_col].transform(function)' to predict {y_attr}.\
#             Select the set of groupby_col and agg_col and also the aggrgate function. \
#             Note that setting {y_attr} as agg_col might be helpful. \n{format_instructions}",
#             input_variables=["data_prompt", "y_attr"],
#             partial_variables={"format_instructions": self.format_instructions}
#         )
#         chain = LLMChain(llm=llm, prompt=prompt)
#         result_lst = []
#         for i in range(n):
#             run  =chain.run({"data_prompt": data_prompt, "y_attr":self.y_attr})
#             print(run)
#             result_lst.append(run)
#         print(result_lst)
#         for result_str in result_lst:
#             print(self.y_attr)
#             print(result_str)
#             try:
#                 result_dict = self.output_parser.parse(result_str)
#                 result_dict['new_feature'] = 'GROUPBY_' + str(result_dict['groupby_col']) + '_' + str(result_dict['function']) + '_' + str(result_dict['agg_col'])
#                 res_lst.append(result_dict)
#             except: 
#                 print("result cannot parse")
#                 continue
#         print("The result list for aggregator is:")
#         print(res_lst)
#         return res_lst

    
class MultiExtractor(Operator):
    def __init__(self, agenda, model_prompt, y_attr, org_attrs):
        super().__init__(agenda, model_prompt, y_attr)
        self.org_attrs = org_attrs
    def __str__(self):
        return f"Attribute description: '{self.data_agenda}', multiextract a reasonable new attribute for predicting '{self.y_attr}' using {str(self.org_attrs)}."
    def generate_new_feature(self, temp, n):
        data_prompt = self.data_agenda
        op_prompt = extractor_prompt_cot.format(y_attr= self.y_attr)
        prompt = "Attribute description: " + data_prompt +'\n'+ op_prompt
        answer = gpt_cot_extract(prompt = prompt, n = 1, temperature=temp)
        try:
            res_lst = [] 
            answer_dict = eval(answer[0])
            print(answer_dict)
            res_lst.append(answer_dict)
        except:
            print("result cannot parse")
            res_lst.append(None)
        return res_lst
    def find_function(self, result_dict, temp=0.3):
        new_col, descr, rel_cols, rel_agenda = self.parse_output(result_dict)
        prompt = extractor_function_prompt_cot.format(data_prompt= rel_agenda, new_feature= new_col, relevant= rel_cols, description= descr)
        # print("prompt")
        # print(prompt)
        answer = gpt_cot_extract(prompt = prompt, n = 1, temperature=temp)[0]
        print("=========================")
        print(answer)
        print("=========================")
        if 'NEED' in answer:
            print("Need to use text completion.")
            return 'TEXT'
        elif 'Cannot' in answer:
            print("Cannot find a function or use text completion.")
            return None
        else:           
            if 'EXTERNAL' in answer:
                print("External sources needed")
                print(answer)
                return None
            else:
                pattern = r"```python(.*?)```"
                match = re.search(pattern, answer, re.DOTALL)
                code = match.group(1).strip()
                return code            
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = result_dict['relevant']
        rel_agenda = obtain_rel_agenda(rel_cols, self.data_agenda)
        return new_col, descr, rel_cols, rel_agenda