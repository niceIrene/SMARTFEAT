import os
import openai
openai.organization = "org-phrO7Hku3zK9ClYOL9ysoBkI"
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
import re
import sys
sys.path.append('../')
sys.path.append("./")
sys.path.append('../prompts/')
from prompt import system_message_prompt, system_message_agg_prompt, system_message_extract_prompt

def completions(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt_fix_or_propose(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    # print(prompt)
    print(response.choices[0].message['content'])
    result_lst = response.choices[0].message['content'].split('\n')
    return result_lst

def split_proposal(text):
    split_text = re.split(r'\b[hH]\d+\.\s*', text)
    return [item.strip() for item in split_text if item.strip()]

def gpt_fix_or_propose_binary(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    # returns the optimal choice
    # print(prompt)
    return_str = response.choices[0].message['content'].replace('\n', '').strip()
    print(return_str)
    result_lst = split_proposal(return_str)
    print(result_lst)
    return result_lst



# def gpt_cot_binary(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None):
#     messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
#     response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,n=n, stop=stop)
#     # returns the optimal choice
#     result_lst = []
#     for c in response.choices:
#         res_str = c.message['content']
#         res_dic = eval(res_str)
#         result_lst.append(res_dic)
#     # print(result_lst)
#     return result_lst

def gpt_cot_binary(prompt, model="gpt-4", temperature=0.7, max_tokens=300, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,n=n, stop=stop)
    # returns the optimal choice
    result_lst = []
    for c in response.choices:
        res_str = c.message['content']
        result_lst.append(res_str)
    # print(result_lst)
    return result_lst


def gpt_cot_extract(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_extract_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,n=n, stop=stop)
    # returns the optimal choice
    result_lst = []
    for c in response.choices:
        result_lst.append(c.message['content'])
    print(result_lst)
    return result_lst

def gpt_cot_agg(prompt, model="gpt-4", temperature=0.7, max_tokens=300, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_agg_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,n=n, stop=stop)
    # returns the optimal choice
    # print(response.choices[0].message['content'])
    result_lst = []
    for c in response.choices:
        res_str = c.message['content']
        result_lst.append(res_str)
    print(result_lst)
    return result_lst