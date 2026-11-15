from model.training_setter import CustomModel
import torch
import torch.nn as nn
from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights
from torchvision.models.convnext import LayerNorm2d


class ModelBasedConvNeXtBase(CustomModel):

    def __init__(self, num_classes: int):
        super().__init__()
        self.model_name = "convnext_based"
        self.features = None
        self.classifier = None

        self._set_model(num_classes)

    def _set_model(self, num_classes: int):
        base_model = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1,
            stochastic_depth_prob=0.2, # 무작위 레이어 스킵
        )
        base_model.classifier = nn.Identity()

        self.features = base_model
        self.classifier = nn.Sequential(
            LayerNorm2d(1024, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(1024, num_classes),
        )

    def set_grad(self, features_freeze: bool):
        for param in self.features.parameters():
            param.requires_grad = False if features_freeze else True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        outputs = self.classifier(features) # (batch_size, n_class)
        
        return outputs
