python prediction_new.py \
    --path ../dataset/cali_housing/ \
    --predict_col BinaryHouseVal \
    --csv housing_cat.csv \
    --model gpt-3.5-turbo \
    --temperature 0.7 \
    --n_generate_sample 10 \
    --clf_model DecisionTree \
    --feature_selection 100  \
    --delimiter 1  \
    --sampling_budget 10
    ${@}