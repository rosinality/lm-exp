#!/bin/bash

sudo apt update
sudo apt install -y python3.10-venv

python3.10 -m venv openmoe_venv

source openmoe_venv/bin/activate
python3 -m pip install -U pip setuptools wheel ipython
python3 -m pip install --upgrade pip

pip install 'transformers[flax]'
pip install datasets
pip install tensorflow-cpu tf-keras

pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install git+https://github.com/google-deepmind/optax
pip install -U flax

sed -i '270aforce_fp32_for_softmax=True,' openmoe_venv/lib/python3.10/site-packages/transformers/models/gpt2/modeling_flax_gpt2.py

python run_clm_flax.py \
    --output_dir="./owm" \
    --model_type="gpt2" \
    --config_name="./owm" \
    --tokenizer_name="./owm" \
    --dataset_name="/mnt/disks/persist/dataset" \
    --cache_dir="/mnt/disks/persist" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size="64" \
    --learning_rate=$LR --warmup_steps="1000" \
    --optimizer=$OPTIMIZER \
    --dtype="bfloat16" \
    $ARGS \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="10" \
    --logging_steps=500 \
    --save_steps="10000" \
    --eval_steps="10000000000000" \
    --exp_name="$OPTIMIZER-$LR-$NAME"