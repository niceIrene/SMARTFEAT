python prediction_new.py \
    --path ../dataset/heart/ \
    --predict_col TenYearCHD \
    --csv framingham_clean.csv \
    --model gpt-3.5-turbo \
    --temperature 0.9 \
    --n_generate_sample 10 \
    --clf_model DecisionTree \
    --feature_selection 100  \
    --delimiter 1  \
    --sampling_budget 10
    ${@}