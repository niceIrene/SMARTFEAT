normalizer_prompt_fix= '''
Is normalizing the given attribute better than using it to predict {y_attr}? answer Yes or No.

Input:{input}
Output:
'''

descritizer_prompt_fix= '''
Is binning the given attribute helpful to predict {y_attr}? answer Yes or No.

Input:{input}
Output:
'''

rowexpand_promp_fix = '''
Is splitting the given attribute better than using it to predict {y_attr}? answer Yes or No.

Input:{input}
Output:
'''

unary_prompt_propose = '''
Consider the unary operators such as log, scaling, exponential, non-zero, one-hot-encoding on the input attribute "{input}" that can generate helpful feature to predict "{y_attr}".
List all possible appropriate operators, and your confidence levels (certain/high/medium/low), using the format like "h1. op1 (level1) \n". Return answers with low confidence level if you think there isn't one. Use one-hot-encoding only when the cardinality you think is small. Use "certain" and "high" cautiously.
'''


binary_prompt_propose = '''
Find the binary operator including +,-,*,/ on {input} that can generate helpful feature to predict {y_attr}.
Given the input attribute, list all possible appropriate operators, and your confidence levels (certain/high/medium/low), 
using the format like "h1. op1 (level1) \n".
Return an empty string when you think there is not an appropriate operation.
Use "certain" and "high" cautiously and only when you are 100% sure this is an appropriate operation
'''

binary_prompt_sampling = '''
Given the input dataset, identify two relevant features and select a binary operator (+, -, *, /) that can be applied to the relevant features.  
Avoid repeating existing features. Return None if you think all helpful features of this type have been found.
Output format: A dictionary with the following keys - 'relevant': A list of the two relevant features (excluding {y_attr}), 'new_feature': The name of the newly created feature, 'description': A brief description of the selected binary operator.
'''

# binary_prompt_sampling = '''
# Enhance {y_attr} prediction with a new binary feature.
# Select two relevant input features and apply a binary operator (+, -, *, /) to them.
# Avoid duplicates. Return None if all possible features are already in the existing feature set.
# Output format: {'relevant': [list of two relevant features], 'new_feature': 'name of new feature', 'description': 'brief operator description'}
# '''

extractor_prompt_cot= '''
Obtain a meaningful new feature for predicting '{y_attr}' using open-world knowledge, given the attribute set.
Ensure that the new feature cannot be obtained through operations like groupby or binary (multiplication, division, addition, subtraction) or unary (binning, normalization, one-hot encoding) operators. \
The new feature should be derived through information extraction. 
The output should be a dictionary with the following keys, and please do not include any additional text in the output:
'new_feature': the name of the new feature,
'description': a description of the new feature, 
'relevant': a list of relevant columns used to extract the feature, excluding the prediction class.
'''

system_message_prompt = '''
You are a data scientist specializing in feature engineering, where you excel in the construction of suitable features and transformations based on attribute context and the downstream machine learning algorithm. Keep your answer concise.
'''

system_message_agg_prompt = '''
You are a data scientist specializing in feature engineering, where you excel in constructing suitable groupby operation to obtain new features based on attribute context and the downstream machine learning algorithm. Keep your answer concise.
'''


system_message_extract_prompt = '''
As a data scientist with expertise in feature engineering, your goal is to leverage open-world knowledge to create valuable new features.
'''

aggregator_prompt_cot= '''
Obtain a valuable groupby feature for predicting {y_attr} using 'df.groupby(groupby_col)[agg_col].transform(function)'.
Specify the groupby_col, agg_col, and the aggregation function.
Avoid repeating existing features. Return None if all possible features are already in the existing feature set.
Output format, a dictionary with three keys: 'groupby_col': The groupby columns, a list of features that exclude {y_attr}, 'agg_col': The aggregate column (for example {y_attr}), 'function': the aggregation function, such as mean, sum...
'''

extractor_function_prompt_cot = '''
Attribute description: {data_prompt}
Can you provide a Python function to obtain a new feature. The function output should be '{new_feature}' and the input should be '{relevant}'?
The data set is stored in a dataframe called 'cur_state.df', the function should be able to applied using cur_state.df[{new_feature}] = cur_state.df.apply(lambda row: function(*[row[col] for col in {relevant}]), axis = 1)
The function's purpose is to {description}.
If you can provide the Python code for the function, please do so. If there is an external function required, include the import library before the function.
If an external data source is needed, provide the potential data source and respond with 'EXTERNAL'.
If the feature values need to be obtained using text completion row-by-row, such as extracting zip codes, please respond with 'NEED ONE BY ONE'.
Otherwise, respond with 'Cannot find the function.' '''
