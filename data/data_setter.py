import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):

    def __init__(
        self, 
        table: pd.DataFrame, 
        image_path_column: str, 
        label_columns: list[str], 
        transforms: transforms,
    ):
        self.table = table
        self.transforms = transforms

        self.image_paths = self.table[image_path_column]
        self.labels_array = self.table[label_columns].values.astype(np.float32)
        
    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        image_path = self.image_paths.iloc[idx]
        image = self.transforms(image_path)
        
        labels = self.labels_array[idx] 
        labels = torch.tensor(
            labels, dtype=torch.float32 # float type: criterion = BCEWithLogitsLoss
        ) 
        return image, labels


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DataSetterConfig:

    images_dir: str
    chexpert_label_list: list[str]
    chexpert_labels_path: str
    seed: Any
    image_transform_set: dict[str, transforms.Compose]
    batch_size: int
    num_workers: int


class DataSetter:

    def __init__(self, data_setter_config: DataSetterConfig):
        self.data_setter_config = data_setter_config

    def load_raw_labels(self, chexpert_labels_path: str) -> pd.DataFrame:
        return pd.read_csv(chexpert_labels_path)

    def preprocess_labels(
        self, labels: pd.DataFrame, images_dir: str, chexpert_label_list: list[str]
    ) -> pd.DataFrame:
        labels["subject_id"] = labels["subject_id"].astype("str")
        labels["study_id"] = labels["study_id"].astype("str")
        labels["dicom_id"] = labels["dicom_id"].astype("str")

        key_1 = "p" + labels["subject_id"].str[:2]
        key_2 = "p" + labels["subject_id"]
        key_3 = "s" + labels["study_id"]
        image_name = labels['dicom_id'] + ".jpg"

        labels["image_path"] = (
            images_dir + "/" + key_1 + "/" + key_2 + "/" + key_3 + "/" + image_name
        )
        ordered_cols = ["subject_id", "image_path"] + chexpert_label_list
        return labels[ordered_cols]

    def split_labels(
        self, labels: pd.DataFrame, seed: Any
    ) -> dict[str, pd.DataFrame]:
        all_ids = labels["subject_id"].unique()

        train_ids, test_ids = train_test_split(
            all_ids, test_size=0.2, random_state=seed, shuffle=True
        )
        train_ids, validation_ids = train_test_split(
            train_ids, test_size=0.25, random_state=seed, shuffle=True
        )
        return {
            "train_labels": labels[labels["subject_id"].isin(train_ids)], 
            "validation_labels": labels[labels["subject_id"].isin(validation_ids)], 
            "test_labels": labels[labels["subject_id"].isin(test_ids)], 
        }
        
    def get_data_loader(
        self, 
        labels: pd.DataFrame, 
        chexpert_label_list: list[str],
        image_transform: transforms.Compose,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        return DataLoader(
            CustomDataset(
                table=labels, 
                image_path_column="image_path",
                label_columns=chexpert_label_list,
                transforms=image_transform,                
            ),
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True if num_workers else False,
        )

    def get_data_loaders(
        self, 
        labels_set: dict[str, pd.DataFrame],
        chexpert_label_list: list[str],
        transform_set: dict[str, transforms.Compose],
        batch_size: int, 
        num_workers: int, 
    ) -> dict[str, DataLoader]:
        train_loader = self.get_data_loader(
            labels=labels_set["train_labels"],
            chexpert_label_list=chexpert_label_list,
            image_transform=transform_set["train_transform"],
            batch_size=batch_size,
            num_workers=num_workers,
        )
        validation_loader = self.get_data_loader(
            labels=labels_set["validation_labels"],
            chexpert_label_list=chexpert_label_list,
            image_transform=transform_set["test_transform"],
            batch_size=batch_size,
            num_workers=num_workers,
        )
        test_loader = self.get_data_loader(
            labels=labels_set["test_labels"],
            chexpert_label_list=chexpert_label_list,
            image_transform=transform_set["test_transform"],
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return {
            "train_loader": train_loader, 
            "validation_loader": validation_loader, 
            "test_loader": test_loader,
        } 

    def work(self, test: bool = False) -> dict[str, DataLoader]:
        load_raw_labels_config = {
            "chexpert_labels_path": self.data_setter_config.chexpert_labels_path
        }
        labels = self.load_raw_labels(**load_raw_labels_config)

        preprocess_labels_config = {
            "labels": labels, 
            "images_dir": self.data_setter_config.images_dir, 
            "chexpert_label_list": self.data_setter_config.chexpert_label_list, 
        }
        labels = self.preprocess_labels(**preprocess_labels_config)
        if test:
            labels = labels.iloc[:100]

        split_labels_config = {
            "labels": labels, 
            "seed": self.data_setter_config.seed, 
        }
        labels_set = self.split_labels(**split_labels_config)

        get_data_loaders_config = {
            "labels_set": labels_set, 
            "chexpert_label_list": self.data_setter_config.chexpert_label_list, 
            "transform_set": self.data_setter_config.image_transform_set, 
            "batch_size": self.data_setter_config.batch_size, 
            "num_workers": self.data_setter_config.num_workers, 
        }
        loader_set = self.get_data_loaders(**get_data_loaders_config)

        return {
            "train_loader": loader_set["train_loader"], 
            "validation_loader": loader_set["validation_loader"], 
            "test_loader": loader_set["test_loader"], 
        }
