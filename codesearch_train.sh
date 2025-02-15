python3 main_codeSearch.py \
    --output_dir=./ \
    --train_data_file_CodeSearch=./datasets/code_search/train.jsonl \
    --eval_data_file_CodeSearch=./datasets/code_search/valid.jsonl \
    --test_data_file_CodeSearch=./datasets/code_search/test.jsonl \
    --model_name_or_path=microsoft/unixcoder-base  \
    --tokenizer_name=microsoft/unixcoder-base  \
    --nl_length 128 \
    --code_length 512 \
    --do_train True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --train_data_rate_code_search 1.0 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --num_train_epochs 10 \
    --seed 42 2>&1 | tee ./topConfigs/defect_to_codesearch.log

    