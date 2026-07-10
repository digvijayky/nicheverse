"""nicheverse.training: the Trainer, training loop, and inference."""

from __future__ import annotations

from .metrics import codebook_usage_stats, plot_training_curves, write_metrics_csv
from .predict import predict_codes
from .pretrain import mae_pretrain
from .trainer import TrainConfig, Trainer, train_model

__all__ = [
    "Trainer",
    "TrainConfig",
    "codebook_usage_stats",
    "mae_pretrain",
    "plot_training_curves",
    "predict_codes",
    "train_model",
    "write_metrics_csv",
]
