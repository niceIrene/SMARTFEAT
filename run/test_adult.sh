python prediction_new.py \
    --path ../dataset/adult/ \
    --predict_col income \
    --csv adult.csv \
    --model gpt-3.5-turbo \
    --temperature 0.9 \
    --n_generate_sample 10 \
    --clf_model LogisticRegression \
    --delimiter 1  \
    --sampling_budget 10
    ${@}