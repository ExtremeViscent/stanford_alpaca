#!/bin/bash

deepspeed --include localhost:0 train.py \
    --model_name_or_path facebook/opt-350m \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir outputs/llama-7b-hf \
    --num_train_epochs 3 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --virtual_batch True \
    --sub_batch_size 32

deepspeed --include localhost:1 --master_port 11452 train.py \
    --model_name_or_path facebook/opt-350m \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir outputs/llama-7b-hf \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --virtual_batch False \
    --sub_batch_size 32