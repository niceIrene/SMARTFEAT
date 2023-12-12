python prediction_new.py \
    --path ../dataset/tennis/ \
    --predict_col Result \
    --csv tennis_clean.csv \
    --model gpt-3.5-turbo \
    --temperature 0.7 \
    --n_generate_sample 10 \
    --clf_model DecisionTree \
    --delimiter 1  \
    --sampling_budget 10
    ${@}