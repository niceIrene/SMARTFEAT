 # SMARTFEAT: Efficient Feature Construction through Feature-Level Foundation Model Interactions
 Raw data collected through data integration is seldom suitable for direct use for such machine learning (or any other data analytics): there is typically a need for appropriate data wrangling to construct high-quality features. This process is highly dependent on domain expertise and requires considerable manual effort by data scientists
 
 In this repo, we introduce SMARTFEAT, an efficient automated feature engineering tool to assist data users, even nonexperts, in constructing useful features. Leveraging the power of Foundation Models (FMs), our approach enables the creation of new features from the data, based on contextual information and open-world knowledge.
 
If you have any questions, please contact: Yin Lin (irenelin@umich.edu)




## Prerequisites
### Step 1: To install the packages and prepare for use, run:
```bash
$ pip install -r requirements.txt
```

### Step 2: Openai configurations
Configure your OpenAI API keys by referring to the guidance available at this link: [Best Practices for API Key Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).
Make sure to replace your openai API key in line 3-5 in file gpt.py
```python
openai.organization = "[YOUR_ORG]"
openai.api_key = "[YOUR OPENAI_API_KEY]"
os.environ["OPENAI_API_KEY"] = "[YOUR OPENAI_API_KEY]"
```


## Example of running pima dataset

```bash
$ cd test
$ ./test_pima.sh
```

We provide an example of the new features for downstream Decision Tree model in ./dataset/pime_example.csv.
The evaluation file for seeing the feature importance and prediction results is in ./baseline/determine_useful_attrs_classication.py

## Running your own dataset

You can add the source file and the corresponding feature description. See examples in the /dataset/

To ensure the performance of the models, make sure to clean the input data as it (1) does not contain NULL values (2) does not have errors.