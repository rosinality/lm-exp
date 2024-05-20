#!/bin/bash

source openmoe_venv/bin/activate

python run_clm_flax.py \
    --output_dir="./owm" \
    --model_type="gpt2" \
    --config_name="./owm" \
    --tokenizer_name="./owm" \
    --dataset_name="/home/rosinality/lm-exp-dataset" \
    --cache_dir="/home/rosinality/bucket" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size="64" \
    --learning_rate=$LR --warmup_steps=$WARMUP \
    --optimizer=$OPTIMIZER \
    --dtype="bfloat16" \
    $ARGS \
    --weight_decay=$WEIGHT_DECAY \
    --overwrite_output_dir \
    --num_train_epochs="2" \
    --logging_steps=500 \
    --save_steps="10000000000000" \
    --eval_steps="10000000000000" \
    --exp_name="$OPTIMIZER-$LR-$WEIGHT_DECAY-$ARGS"