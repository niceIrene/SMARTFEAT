import os
import openai
openai.organization = "YOUR_ORG"
openai.api_key = "YOUR_OPENAI_APIKEY"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_APIKEY"

import re
import sys
sys.path.append('../')
sys.path.append("./")
sys.path.append('../prompts/')
from prompt import system_message_prompt

def completions(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# can use 3.5-turbo for better efficiency
def gpt_propose(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    result_lst = response.choices[0].message['content'].split('\n')
    print(result_lst)
    return result_lst

# proposal approach for binary
def gpt_propose_binary(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    # returns the optimal choice
    return_str = response.choices[0].message['content'].replace('\n', '').strip()
    result_lst = split_proposal(return_str)
    print(result_lst)
    return result_lst


def gpt_sampling(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,n=n, stop=stop)
    result_lst = []
    for c in response.choices:
        res_str = c.message['content']
        result_lst.append(res_str)
    print(result_lst)
    return result_lst

# 3.5 has better performance than 4
def gpt_sampling_extract(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'system', 'content': system_message_prompt}, {"role": "user", "content": prompt}]
    response = completions(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,n=n, stop=stop)
    result_lst = []
    for c in response.choices:
        result_lst.append(c.message['content'])
    print(result_lst)
    return result_lst