import dataclasses
from typing import Any

import pandas as pd
import torch
import torch.nn as nn


class CustomModel(nn.Module):

    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainingSetterConfig:

    model_name: str
    weight_decay: float
    warm_up: bool
    train_labels: pd.DataFrame
    chexpert_label_list: list[str]
    epochs: int
    minimum_scheduler_learning_rate: float


class TrainingSetter:

    def __init__(self, training_setter_config: TrainingSetterConfig):
        self.training_setter_config = training_setter_config

    def select_model(self, model_name: str, num_classes: int) -> CustomModel:
        if model_name == "convnext_base":
            from model.models.convnext_base import ModelBasedConvNeXtBase
            return ModelBasedConvNeXtBase(num_classes)
        raise ValueError(f"Do not support model: {model_name}")

    def get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_optimizer(
        self, model: CustomModel, weight_decay: float, warm_up: bool
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=self._set_param_groups(model, warm_up), 
            weight_decay=weight_decay,
        )

    def _set_param_groups(self, model: CustomModel, warm_up: bool) -> list[dict[str, Any]]:
        if model.model_name == "convnext_based":
            param_groups = [{'params': model.classifier.parameters(), 'lr': 1e-3}]
            if not warm_up:
                param_groups.append({'params': model.features.parameters(), 'lr': 1e-5})
        return param_groups

    def get_criterion(
        self, train_labels: pd.DataFrame, device: str, chexpert_label_list: list[str]
    ) -> nn.modules.loss._Loss:
        return nn.BCEWithLogitsLoss(
            pos_weight=self._set_pos_weights(
                train_labels, chexpert_label_list
            ).to(device), 
            reduction='none',
        )
    
    def _set_pos_weights(
        self, train_labels: pd.DataFrame, chexpert_label_list: list[str]
    ) -> torch.Tensor:
        num_pos = (train_labels[chexpert_label_list] == 1.0).sum(axis=0)
        num_neg = (train_labels[chexpert_label_list] == 0.0).sum(axis=0)

        pos_weights = num_neg / (num_pos + 1e-6)
        pos_weights = torch.tensor(pos_weights.values, dtype=torch.float)

        return pos_weights.clamp(min=0.1, max=10.0)

    def get_scheduler(
        self, 
        optimizer: torch.optim.Optimizer, 
        epochs: int, 
        minimum_scheduler_learning_rate: float,
    ) -> torch.optim.lr_scheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=minimum_scheduler_learning_rate
        )

    def work(self) -> dict[str, Any]:
        select_model_config = {
            "model_name": self.training_setter_config.model_name, 
            "num_classes": len(self.training_setter_config.chexpert_label_list), 
        }
        model = self.select_model(**select_model_config)

        get_optimizer_config = {
            "model": model,  
            "weight_decay": self.training_setter_config.weight_decay,  
            "warm_up": self.training_setter_config.warm_up,  
        }
        optimizer = self.get_optimizer(**get_optimizer_config)

        device = self.get_device()
        get_criterion_config = {
            "train_labels": self.training_setter_config.train_labels, 
            "device": device, 
            "chexpert_label_list": self.training_setter_config.chexpert_label_list, 
        }
        criterion = self.get_criterion(**get_criterion_config)

        get_scheduler_config = {
            "optimizer": optimizer, 
            "epochs": self.training_setter_config.epochs, 
            "minimum_scheduler_learning_rate": 
                self.training_setter_config.minimum_scheduler_learning_rate, 
        }
        scheduler = self.get_scheduler(**get_scheduler_config)

        return {
            "model": model, 
            "optimizer": optimizer, 
            "device": device, 
            "criterion": criterion, 
            "scheduler": scheduler, 
        }
