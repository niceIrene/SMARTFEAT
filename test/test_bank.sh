python prediction_new.py \
    --path ../dataset/bank/ \
    --predict_col predict_col \
    --csv bank.csv \
    --model gpt-3.5-turbo \
    --temperature 0.7 \
    --n_generate_sample 10 \
    --clf_model DecisionTree \
    --feature_selection 100  \
    --delimiter 2  \
    --sampling_budget 10
    ${@}