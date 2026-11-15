import dataclasses
import os
import pathlib
from PIL import Image

from torchvision import transforms


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Config:

    environment = {
        "seed": 42,
    }

    path = {
        "images": pathlib.Path("/datasets/MIMIC-CXR/images"), 
        "chexpert_labels_path": pathlib.Path("/datasets/MIMIC-CXR/label_chexpert.csv"),
    }

    label = {
        "chexpert_label_list": [
            "Atelectasis", 
            "Cardiomegaly", 
            "Consolidation", 
            "Edema", 
            "Enlarged Cardiomediastinum", 
            "Fracture", 
            "Lung Lesion", 
            "Lung Opacity", 
            "Pleural Effusion", 
            "Pleural Other", 
            "Pneumonia", 
            "Pneumothorax", 
            "Support Devices", 
        ],
    }

    image = {
        "train_transform": transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path).convert("RGB")),
            transforms.Resize((384, 384)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ]),
        "test_transform": transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path).convert("RGB")),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ]),
    }

    model = {
        "common": {
            "model_parameter_dir": pathlib.Path(
                "/home/jovyan/students/class_deep_learning/21016046"
            ),
        },
        "convnext_base": {
            "model_parameter_file_name": pathlib.Path("convnext_base.pth"), 
            "warm_up": True, 
            "warm_up_epochs": 5, 
            "training_epochs": 10, 
            "batch_size": 32, 
            "weight_decay": 5e-2, 
            "minimum_scheduler_learning_rate": 1e-6, 
            "num_workers": 2, 
        },
    }


os.environ["TORCH_HOME"] = str(Config().model["common"]["model_parameter_dir"])
