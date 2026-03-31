#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

python -m train_vae_mnist --cfg_path mnist_ldce/configs/autoencoder_mnist_kl.yaml \
                          --output_dir mnist_ldce/checkpoints/mnist_vae \
                          --data_root  mnist_ldce/data
