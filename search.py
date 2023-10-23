import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from operator_new import Descrtizer, Normalizer, RowExpand, BinaryOperator, MultiExtractor, AggregateOperator, UnaryOperator, BinaryOperatorAlter
from serialize import *
import copy
import itertools
import pandas as pd


class CurrentAttrLst(object):
    # maintain a set of current attributes with a similarity checker
    def __init__(self, cur_attr_lst, agenda, data_df, model, budget):
        self.cur_attr_lst =copy.deepcopy(cur_attr_lst)
        self.cur_attr_str = str(cur_attr_lst)
        self.data_agenda = agenda
        self.step = 0
        self.last_op = None
        self.df = data_df
        self.model = model
        self.previous_two = 0
        self.budget = budget
        self.budget_cur = 0

    def __str__(self):
        return f"The current attribute list is {self.cur_attr_str}"
    def determine_similarity(self, new_attr, new_desr, temp=0.1):
        print("similarity check!!")
        if self.data_agenda.find(new_attr) != -1:
            return True
        else:
            return False
    def update(self, new_attr, new_desr):
        self.cur_attr_lst.append(new_attr)
        self.cur_attr_str = str(self.cur_attr_lst)
        self.data_agenda = self.data_agenda + ", \n {}: {}".format(new_attr, new_desr)
    

def row_serialization(row, attr_lst):
    row_ser = ''
    for a in attr_lst:
        row_ser = row_ser + str(a) + ":" + str(row[a])+ ","
    return row_ser

def text_completion_extract(df, new_feature, temp=0.1):
    llm = OpenAI(temperature = temp, model_name='gpt-3.5-turbo')
    new_col_val = []
    for idx, row in df.iterrows():
        attr_lst = list(df.columns)
        row_str = row_serialization(row, attr_lst)
        try:
            response_schema = [
                ResponseSchema(name=new_feature, description="string or float, representing attribute value"),
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schema)
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template=row_str + "{new_feature}:? \n{format_instructions}",
                input_variables=["new_feature"],
                partial_variables={"format_instructions": format_instructions}
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            result_str = chain.run({"new_feature": new_feature})
            print(result_str)
            result_dict = output_parser.parse(result_str)
            new_value = result_dict[new_feature]
            new_col_val.append(new_value)
        except Exception as e:
            print("Error", str(e))
            new_col_val.append(np.nan)
    df[new_feature] = new_col_val

def feature_generator_propose(cur_state, org_features, predict_col):
    print("The current step is !!!!!")
    print(cur_state.step)
    # Unary
    if cur_state.step < len(org_features):
        col = org_features[cur_state.step]
        if cur_state.last_op is None:
            op_normalize = Normalizer(cur_state.data_agenda, cur_state.model, predict_col, col)
            result_dict = op_normalize.generate_new_feature(temp=0.1)
            cur_state.last_op = op_normalize
            if result_dict is None:
                return result_dict
            else:
                return [result_dict]
        elif isinstance(cur_state.last_op, Normalizer):
            op_descritize = Descrtizer(cur_state.data_agenda, cur_state.model, predict_col, col)
            result_dict = op_descritize.generate_new_feature(temp=0.1)
            cur_state.last_op = op_descritize
            if result_dict is None:
                return result_dict
            else:
                return [result_dict]
        elif isinstance(cur_state.last_op, Descrtizer):
            # obtain a input row
            op_unary = UnaryOperator(cur_state.data_agenda, cur_state.model, predict_col, col)
            res_lst = op_unary.generate_new_feature(temp= 0.1)
            cur_state.last_op = op_unary
            return res_lst
        else:
            return -1
    # binary enumeration
    # elif cur_state.step >= len(org_features):
    #     if cur_state.last_op is None:
    #         pairs = list(itertools.combinations(org_features, 2))
    #         p = pairs[cur_state.step - len(org_features)]
    #         op_binary = BinaryOperator(cur_state.data_agenda, cur_state.model, predict_col, p)
    #         res_lst = op_binary.generate_new_feature(temp= 0.1)
    #         cur_state.last_op = op_binary
    #         return res_lst
    #     elif isinstance(cur_state.last_op, BinaryOperator):
    #         return  -1

    # elif cur_state.step >= len(org_features):
    #     if cur_state.last_op is None:
    #         op_binary = BinaryOperatorAlter(cur_state.data_agenda, cur_state.model, predict_col, org_features)
    #         res_lst = op_binary.generate_new_feature(temp= 0.7)
    #         cur_state.last_op = op_binary
    #         return res_lst
    #     elif isinstance(cur_state.last_op, BinaryOperatorAlter):
    #         return  -1



def feature_genetor_cot(cur_state, predict_col, tempurature, n_sample):
    # for each operator, when we incur three generation failures/ repeated generations, we stop it.
    features = cur_state.cur_attr_lst
    if cur_state.previous_two >2:
        return -1
    if cur_state.last_op is None or (isinstance(cur_state.last_op, BinaryOperatorAlter) and cur_state.previous_two<=2):
        op_binary = BinaryOperatorAlter(cur_state.data_agenda, cur_state.model, predict_col)
        res_lst = op_binary.generate_new_feature(temp= 0.7)
        cur_state.last_op = op_binary
        return res_lst      
    elif isinstance(cur_state.last_op, AggregateOperator) and cur_state.previous_two<=2:
        op_agg = AggregateOperator(cur_state.data_agenda, cur_state.model, predict_col, features)
        res_lst = op_agg.generate_new_feature(temp=tempurature, n = n_sample)
        cur_state.last_op = op_agg
        print(res_lst)
        return res_lst
    elif isinstance(cur_state.last_op, MultiExtractor) and cur_state.previous_two<=2:
        op_multi = MultiExtractor(cur_state.data_agenda, cur_state.model,predict_col, features)
        res_lst = op_multi.generate_new_feature(temp=tempurature, n = 1)
        cur_state.last_op = op_multi
        return res_lst
    

def state_evaluator(result_dict, cur_state, predict_col):
    if cur_state.budget_cur >= cur_state.budget:
        print("Budget reached!!!!")
    if result_dict is None:
        state_update(cur_state, False, "Result Dict is None")
    else:
        if isinstance(cur_state.last_op, Normalizer) or isinstance(cur_state.last_op, Descrtizer) or isinstance(cur_state.last_op, UnaryOperator):
            new_col, descr, rel_cols, rel_agenda = cur_state.last_op.parse_output(result_dict)
            func_str = cur_state.last_op.find_function(result_dict)
            print("!!!!!!!!!!!!!fuction")
            print(func_str)
            one_hot_flag = 0
            if func_str == 'encoding':
                one_hot_flag = 1
            try:
                if one_hot_flag == 0:
                    exec(func_str)
                    func_name =  obtain_function_name(func_str)
                    func_obj =locals()[func_name]
                try:
                    # handle one-hot-encoding as specific case:
                    if 'encoding' in result_dict['description']:
                        org_cols = list(cur_state.df.columns)
                        one_hot_flag = 1
                        onehot_df = pd.get_dummies(cur_state.df, columns=[rel_cols[0]])
                        cur_state.df = pd.concat([cur_state.df[rel_cols[0]], onehot_df], axis=1)
                        new_cols = list(cur_state.df.columns)
                    elif isinstance(cur_state.last_op, Descrtizer):
                        cur_state.df[new_col] = cur_state.df.apply(lambda row: func_obj(row[rel_cols[0]]), axis = 1)
                    else:
                        # cur_state.df[new_col] = func_obj(cur_state.df[rel_cols[0]])
                        cur_state.df[new_col] = func_obj(cur_state.df[rel_cols[0]])  
                    if one_hot_flag:
                        if evaluate_for_one_hot(org_cols, new_cols, 10):
                            state_update(cur_state, True, '',  new_col, descr)
                        else:
                            state_update(cur_state,False, " The cardinality for the one-hot encoding it too big")
                            print("One hot not good!!!!")      
                            print(cur_state.df.columns)
                            cur_state.df = cur_state.df.drop(list(set(new_cols) - set(org_cols)), axis=1)
                    elif ig_evaluate(cur_state.df, new_col) and not one_hot_flag:
                        state_update(cur_state, True, '',  new_col, descr)
                    else:
                        state_update(cur_state,False, "New feature '{}' does not have good information gain.".format(new_col))
                        cur_state.df = cur_state.df.drop([new_col])
                except Exception as e:
                    state_update(cur_state,False, "Function '{}' cannot be applied to the dataframe.".format(func_str))
                    print("Error:", str(e))
            except:
                state_update(cur_state, False, "New Feature: {}, descr: {}, function: {} cannot be obtained or execute".format(new_col, descr, func_str))
        elif isinstance(cur_state.last_op, BinaryOperatorAlter) or isinstance(cur_state.last_op, BinaryOperator):
            new_col = result_dict['new_feature']
            rel_cols = result_dict['relevant']
            descr = result_dict['description']
            func_str = cur_state.last_op.find_function(result_dict)
            print(func_str)
            if cur_state.determine_similarity(new_col, descr):
                state_update(cur_state,False, "New feature '{}' is similar to existing feature".format(new_col))
            else:
                try:
                    exec(func_str)
                    func_name =  obtain_function_name(func_str)
                    func_obj =locals()[func_name]
                    try:
                        cur_state.df[new_col] = cur_state.df.apply(lambda row: func_obj(row[rel_cols[0]], row[rel_cols[1]]), axis = 1)
                        if ig_evaluate(cur_state.df, new_col):
                            cur_state.budget_cur += 1
                            state_update(cur_state, True, '',  new_col, descr)
                        else:
                            state_update(cur_state,False, "New feature '{}' does not have good information gain.".format(new_col))
                            cur_state.df = cur_state.df.drop([new_col])
                    except:
                        state_update(cur_state,False, "Function '{}' cannot be applied to the dataframe.".format(func_str))
                except:
                    state_update(cur_state, False, "New Feature: {}, descr: {}, function: {} cannot be obtained or execute".format(new_col, descr, func_str))
        elif isinstance(cur_state.last_op, AggregateOperator):
            new_col = result_dict["new_feature"]
            groupby_col = result_dict['groupby_col']
            agg_col = result_dict['agg_col']
            function = result_dict['function']
            temp_dict = {}
            for index, r in enumerate(groupby_col):
                if (cur_state.df[r].dtype != 'object') and cur_state.df[r].nunique() > 20:
                    column_lst = list(cur_state.df.columns)
                    if "Bucketized_" + r in column_lst:
                        groupby_col[index] = "Bucketized_" + r
                        continue
                    temp_dict['new_feature'] =  "Bucketized_{}".format(r)
                    temp_dict['description'] =  "Bucketized {}".format(r)
                    temp_dict['relevant'] = r
                    temp_op = Descrtizer(cur_state.data_agenda, cur_state.model,predict_col, r)
                    func_str = temp_op.find_function(temp_dict)
                    print(func_str)
                    exec(func_str)
                    func_name =  obtain_function_name(func_str)
                    func_obj =locals()[func_name]
                    cur_state.df[temp_dict['new_feature']] = cur_state.df.apply(lambda row: func_obj(row[r]), axis = 1)
                    groupby_col[index] = temp_dict['new_feature']
                else:
                    continue
            # print(groupby_col)
            new_col = 'GROUPBY_' + str(groupby_col) + '_' + function + '_' + agg_col
            new_desr = "df.groupby({})[{}].transform({})".format(groupby_col, agg_col, function)
            if cur_state.determine_similarity(new_col, new_desr):
                state_update(cur_state,False, "New feature '{}' is similar to existing feature".format(new_col))
            else:
                cur_state.df[new_col] = cur_state.df.groupby(groupby_col)[agg_col].transform(function)
                if ig_evaluate(cur_state.df, new_col):
                    cur_state.budget_cur += 1
                    state_update(cur_state, True, '',  new_col, new_col)
                else:
                    state_update(cur_state,False, "New feature '{}' does not have good information gain.".format(new_col))
                    cur_state.df = cur_state.df.drop([new_col])
        elif isinstance(cur_state.last_op, MultiExtractor):
            new_col, descr, rel_cols, rel_agenda = cur_state.last_op.parse_output(result_dict)
            try:
                find_answer = cur_state.last_op.find_function(result_dict)
            except:
                state_update(cur_state,False, "find function fail")
                return
            if find_answer is None:
                state_update(cur_state,False, "Invalid feature")
            elif 'TEXT' in find_answer:
                try:
                    text_completion_extract(cur_state.df, new_col)
                except:
                    state_update(cur_state,False, "text-completion error")
            else:
                print("found the lambda function")
                print(find_answer)
                try:
                    exec(find_answer)
                    func_name =  obtain_function_name(find_answer)
                    func_obj = locals()[func_name]
                    if isinstance(rel_cols, list):
                        try:
                            cur_state.df[new_col] = cur_state.df.apply(lambda row: func_obj(*[row[col] for col in rel_cols]), axis = 1)
                            if ig_evaluate(cur_state.df, new_col):
                                cur_state.budget_cur += 1
                                state_update(cur_state, True, '',  new_col, descr)
                            else:
                            # if not success, roll back the udpate
                                state_update(cur_state,False, "New feature '{}' does not have good information gain.".format(new_col))
                                cur_state.df = cur_state.df.drop([new_col])                          
                        except Exception as e:
                            print(e)
                            state_update(cur_state,False, "wrong input format")
                except Exception as e:
                    print(e)
                    state_update(cur_state,False, "Function apply error")            

def is_column_imbalanced(df, column_name, threshold=0.8):
    column_values = df[column_name]
    value_counts = column_values.value_counts()
    most_common_value_count = value_counts.max()
    total_values = len(column_values)
    if most_common_value_count / total_values > threshold:
        return True
    else:
        return False

def ig_evaluate(df, new_col):
    column = df[new_col]
    nan_percentage = column.isna().mean() * 100
    if nan_percentage > 30 or is_column_imbalanced(df, new_col):
        return False
    else:
        return True
    
def evaluate_for_one_hot(org_cols, new_cols, c = 10):
    if len(new_cols) - len(org_cols) >= 10:
        return False
    else:
        return True

def state_update(cur_state, result_f_or_t, false_msg, new_col = '', new_desrc = ''):
    print("The current budget is: ", cur_state.budget_cur)
    if result_f_or_t:
        print("update state")
        cur_state.previous_two = 0
        # update the budget
        cur_state.update(new_col, new_desrc)
        print(cur_state.df.head(10))
        # cache the intermediate result
        cur_state.df.to_csv("current_df_{}_{}.csv".format(cur_state.step, new_col))
    else:
        print(false_msg)
        if isinstance(cur_state.last_op, MultiExtractor) or isinstance(cur_state.last_op, BinaryOperatorAlter) or isinstance(cur_state.last_op, AggregateOperator):
            cur_state.previous_two += 1

