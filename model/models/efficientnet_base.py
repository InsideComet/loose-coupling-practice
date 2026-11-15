from model.training_setter import CustomModel
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights


class ModelBasedEfficientNetV2S(CustomModel):

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = None
        self.classifier = None

        self._set_model(num_classes)

    def _set_model(self, num_classes: int):
        base_model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        base_model.classifier = nn.Identity()

        self.features = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def set_grad(self, features_freeze: bool):
        for param in self.features.parameters():
            param.requires_grad = False if features_freeze else True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        outputs = self.classifier(features) # (batch_size, n_class)
        
        return outputs
