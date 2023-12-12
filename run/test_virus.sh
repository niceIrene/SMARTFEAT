python prediction_new.py \
    --path ../dataset/virus/ \
    --predict_col WnvPresent \
    --csv west_nile.csv \
    --temperature 0.7 \
    --n_generate_sample 10 \
    --clf_model DecisionTree \
    --delimiter 1  \
    --sampling_budget 10
    ${@}