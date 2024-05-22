#!/bin/bash

export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

sudo NEEDRESTART_MODE=a apt-get update -y
sudo NEEDRESTART_MODE=a apt-get install -y python3.10-venv gcsfuse

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
pip install gcsfs==2024.3.1

cp modeling_flax_gpt2.py openmoe_venv/lib/python3.10/site-packages/transformers/models/gpt2/modeling_flax_gpt2.py