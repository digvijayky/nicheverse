#!/usr/bin/env bash
# Reproduce the b32k reference checkpoint behind the Cancer Cell RCC + BrM results
# (173 sample cohort, 5.66M cells). This is the 0.2.0 default configuration:
# batch size 32768, seed 9, spatial graph knn_radius at radius 50 microns,
# k_neighbors 20, learning rate 3e-4, 300 epochs, weighted_mean aggregation.
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
    --spatial-graph knn_radius \
    --radius 50 \
    --aggregation weighted_mean \
    --batch-size 32768 \
    --lr 3e-4 \
    --seed 9
