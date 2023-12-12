python prediction_new.py \
    --path ../dataset/heart/ \
    --predict_col TenYearCHD \
    --csv framingham_clean.csv \
    --temperature 0.9 \
    --n_generate_sample 10 \
    --clf_model DecisionTree \
    --delimiter 1  \
    --sampling_budget 10
    ${@}