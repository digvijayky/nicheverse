#!/usr/bin/env bash
# Reproduce the 173 sample RCC + BrM training run from the Cancer Cell submission.
set -euo pipefail

INPUT=${1:?usage: rcc_brm_train.sh <preprocessed.h5ad> <checkpoint_dir>}
CKPT=${2:?usage: rcc_brm_train.sh <preprocessed.h5ad> <checkpoint_dir>}

nicheverse train \
    --input "$INPUT" \
    --checkpoint-dir "$CKPT" \
    --num-epochs 300 \
    --cell-codebook-size 256 \
    --cell-codebook-embdim 64 \
    --neighborhood-codebook-size 32 \
    --neighborhood-codebook-embdim 256 \
    --k-neighbors 20 \
    --batch-size 2048 \
    --lr 3e-4
