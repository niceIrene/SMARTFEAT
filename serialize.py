import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
import numpy as np

def obtain_rel_agenda(rel_cols, agenda):
    rel_agenda = []
    for col in rel_cols:
        start = agenda.find(col)
        end = agenda.find('\n', start)
        if end == -1:
            end = None
        rel_agenda.append(agenda[start:end])
    return ', '.join(rel_agenda)


def obtain_function_name(func_str):
    match = re.search(r"def (\w+)\(", func_str)
    return match.group(1)


def obtain_relevant_cols(output_dic):
    rel_cols = output_dic["relevant"].split('\n')
    rel_cols_new = []
    for c in rel_cols:
        c = c.strip()
        rel_cols_new.append(c)
    return rel_cols_new
    

def exec_function(func_str, df, new_col, rel_cols):
    start = func_str.find('def')
    func_str = func_str[start:]
    exec(func_str)
    func_name = obtain_function_name(func_str)
    print(func_name, func_str)
    func_obj = globals()[func_name]
    df[new_col] = df.apply(lambda row: func_obj(row[rel_cols[0]], row[rel_cols[1]]), axis = 1)


def row_serialization(row, attr_lst):
    row_ser = ''
    for a in attr_lst:
        row_ser = row_ser + str(a) + ":" + str(row[a])+ ","
    return row_ser


def obtain_function_new_features(func_str):
    pos = func_str.find('return')
    return_str = func_str[pos+7:]
    attributes = return_str.split(',')
    return attributes


def text_completion_extract(df, new_feature, temp=0.1):
    llm = OpenAI(temperature = temp)
    for idx, row in df.iterrows():
        attr_lst = list(df.columns)
        row_str = row_serialization(row, attr_lst)
        response_schema = [
            ResponseSchema(name=new_feature, description="attribute value"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template="Fill in the value for the question mark" + row_str + "{new_feature}:? \n{format_instructions}",
            input_variables=["new_feature"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result_str = chain.run({"new_feature": new_feature})
        print(row_str)
        print(result_str)
        try:
            result_dict = output_parser.parse(result_str)
            new_value = result_dict[new_feature]
        except:
            new_value = np.nan
        row[new_feature] = new_value
