python3 main_clone.py \
    --train_data_file=./datasets/dataset_clone/train.txt \
    --output_dir=./ \
    --eval_data_file=./datasets/dataset_clone/valid.txt \
    --test_data_file=./datasets/dataset_clone/test.txt \
    --model_name_or_path=microsoft/unixcoder-base \
    --tokenizer_name=microsoft/unixcoder-base \
    --num_classes 1 \
    --nl_length 128 \
    --code_length 512 \
    --do_train True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --train_data_rate_clone 0.1 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --num_train_epochs 10 \
    --seed 42 2>&1 | tee ./topConfigs/defect_to_clone.log
