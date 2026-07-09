"""nicheverse.training: the Trainer, training loop, and inference."""

from __future__ import annotations

from .predict import predict_codes
from .pretrain import mae_pretrain
from .trainer import TrainConfig, Trainer, train_model

__all__ = ["Trainer", "TrainConfig", "mae_pretrain", "predict_codes", "train_model"]
