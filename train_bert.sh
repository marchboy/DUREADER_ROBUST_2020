export CUDA_VISIBLE_DEVICES=0
export pretrain_dir=./
export lm=chinese-roberta-wwm-ext
export cache_dir=./cache
export output_dir=./du2020
export task=384_bert


python main_bert.py \
  --model_type bert \
  --summary log/$task \
  --model_name_or_path $pretrain_dir/$lm/pytorch_model.bin \
  --config_name $pretrain_dir/$lm/config.json \
  --tokenizer_name $pretrain_dir/$lm/vocab.txt \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --threads 4 \
  --warmup_ratio 0.1 \
  --logging_ratio 0.1 \
  --save_ratio 0.1 \
  --do_lower_case \
  --overwrite_cache \
  --data_dir ./datasets/dureader_robust-data \
  --train_file train.json \
  --predict_file dev.json \
  --test_file 2020_test1.json \
  --test_prob_file test_prob.pkl \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 32 \
  --max_answer_length 64 \
  --n_best_size 10 \
  --output_dir $output_dir/$task \
  --overwrite_output_dir \
  --do_fgm \
  --gc \
#   --version_2_with_negative \
