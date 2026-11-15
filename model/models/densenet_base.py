from model.training_setter import CustomModel
import torch
import torch.nn as nn
from torchvision.models import densenet121
from torchvision.models import DenseNet121_Weights


class ModelBasedDenseNet121(CustomModel):

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = None
        self.classifier = None

        self._set_model(num_classes)

    def _set_model(self, num_classes: int):
        base_model = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )
        input_features_number = base_model.classifier.in_features
        base_model.classifier = nn.Identity()

        self.features = base_model
        self.classifier = nn.Linear(input_features_number, num_classes)

    def set_grad(self, features_freeze: bool):
        for param in self.features.parameters():
            param.requires_grad = False if features_freeze else True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        outputs = self.classifier(features) # (batch_size, n_class)
        
        return outputs
