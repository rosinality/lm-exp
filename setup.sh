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