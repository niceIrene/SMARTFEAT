unary_prompt_propose = '''
Consider the unary operators such as bucketize, scaling, one-hot-encoding, and rowexpand on the input attribute "{input}" that can generate helpful feature to predict "{y_attr}".
List all possible appropriate operators, and your confidence levels (certain/high/medium/low).
Return answers with low confidence level for less beneficial transformations compared to the original input feature. Use one-hot-encoding only when the cardinality you think is small. Use "certain" and "high" very cautiously.
Output example:
```
h1. one-hot-encoding (high): a short description of feature usefulness \n
```
'''


binary_prompt_propose = '''
Find the binary operator including +,-,*,/ on {input} that can generate helpful feature to predict {y_attr}.
Given the input attribute, list all possible appropriate operators, and your confidence levels (certain/high/medium/low), 
Return an empty string when you think there is not an appropriate operation.
Use "certain" and "high" cautiously and only when you are 100% sure this is an appropriate operation.
Output example:
```
h1. division (medium): a short description of feature usefulness \n
```
'''

binary_prompt_sampling = '''
Given the input dataset, identify two relevant features and select a binary operator (+, -, *, /) that can be applied to the relevant features.  
Choose the ones that you believe will be useful for a downstream classification algorithm. 
Output format: A dictionary with the following keys - 
'new_feature': the name of the new feature,
'description': a description of the new feature, 
'relevant': a list of the two relevant features (excluding {y_attr}).
'''

extractor_prompt_sampling= '''
Generate a meaningful new feature for predicting {y_attr} using open-world knowledge and the attribute set.
Employ methods such as feature combinations, knowledge extrations, and excluding existing operators such as binary and unary operations.
The new feature should be derived through information extraction. 
Output format: A dictionary with the following keys - 
'new_feature': the name of the new feature,
'description': a description of the new feature, 
'relevant': a list of relevant columns used to extract the feature (excluding {y_attr}).
'''

system_message_prompt = '''
You are an expert datascientist working to improve predictions. 
You perform data transformations that generate additional columns that are useful for a downstream classification algorithm.
Please closely follow the provided column descriptions, considering data types and ranges.
'''

aggregator_prompt_sampling= '''
Obtain a valuable groupby feature for predicting {y_attr} using 'df.groupby(groupby_col)[agg_col].transform(function)'.
Specify the groupby_col, agg_col, and the aggregation function.
Choose the ones that you believe will be useful for a downstream classification algorithm. 
Output format, a dictionary with the following keys -
'groupby_col': The groupby columns, a list of features that exclude {y_attr},
'agg_col': The aggregate column (for example {y_attr}),
'function': the aggregation function, such as mean, sum...
'''

extractor_function_prompt = '''
Generate the most appropriate python function to obtain new feature {new_feature} (output) using features {relevant} (input), new feature description: {description}, input feature description: {rel_agenda}.
Please closely follow the provided column descriptions, considering data types and ranges.
Consider the following situiations:
(1) If you can provide the Python code for the function, please do so. If an external function is required, include the import library before the function. 
The data set is stored in a dataframe called 'cur_state.df', the function should be able to applied using cur_state.df[{new_feature}] = cur_state.df.apply(lambda row: function(*[row[col] for col in {relevant}]), axis = 1)
Code format:

```python
# Import necessary libraries:
import xxx
# Function definition
def function_name(xxx):
    xxx
    return xxx
```end

(2) If an external data source is needed, provide details about the potential data source and respond with 'EXTERNAL'.
(3) If the feature values must be obtained using text completion row-by-row (e.g., extracting zip codes), respond with 'NEED ONE BY ONE'.
(4) Otherwise, respond with 'Cannot find the function.' 
'''
