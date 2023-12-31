python prediction_new.py \
    --path ../dataset/bank/ \
    --predict_col predict_col \
    --csv bank.csv \
    --temperature 0.7 \
    --n_generate_sample 10 \
    --clf_model DecisionTree \
    --delimiter 2  \
    --sampling_budget 10
    ${@}