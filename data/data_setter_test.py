from PIL import Image

from config import Config
from data.data_setter import DataSetter
import pandas as pd
import pytest
import torch


_CONFIG = Config()
_ENVIRONMENT = _CONFIG.environment
_PATH = _CONFIG.path
_LABEL = _CONFIG.label
_IMAGE = _CONFIG.image
_MODEL = _CONFIG.model

TEST_MODEL_NAME = "convnext_base"
_TEST_MODEL = _MODEL[TEST_MODEL_NAME]


class TestCheXpert:
    """Test CheXpert label dataset."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.ids = ["dicom_id", "subject_id", "study_id"]
        self.label_dataset = pd.read_csv(_PATH["chexpert_labels_path"])
        self.label_list = _LABEL["chexpert_label_list"]

    def test_dataset_columns(self):
        """Check the origin labels based on `docs/DeepLearning2_Project.pdf`"""
        correct_columns = set(
            self.ids + self.label_list + ["No Finding"] # check `README.md` # 1. 개요
        )
        assert correct_columns == set(self.label_dataset.columns)

    def test_dataset_values(self):
        """Check if all dataset values are valid."""
        correct_label_values = [1.0, 0.0, -1.0] # not consider NaN because can't compare
        dataset_values = self.label_dataset[self.label_list].values.flatten()
        dataset_values = pd.unique(
            dataset_values[pd.notna(dataset_values)]
        )
        assert set(correct_label_values) == set(dataset_values)

    def test_dataset_ids_missing_values(self):
        """Check if there are any missing values in id columns."""
        for id in self.ids:
            id_values = self.label_dataset[id]
            assert sum(pd.isna(id_values)) == 0

    def test_dataset_ids_uniqueness(self):
        """Check if each row's id columns combination is unique."""
        assert not self.label_dataset.duplicated(self.ids).any()


class TestDataSetter:
    """Check the functions related to data handling."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.data_setter = DataSetter(data_setter_config=None)
        self.chexpert_label_list = _LABEL["chexpert_label_list"]
        self.raw_labels = self.data_setter.load_raw_labels(_PATH["chexpert_labels_path"])
        self.images_dir = _PATH["images"]
        self.preprocessed_labels = self.data_setter.preprocess_labels(
            self.raw_labels, str(self.images_dir), self.chexpert_label_list
        )
        self.seed = _ENVIRONMENT["seed"]
        self.image_test_transform = _IMAGE["test_transform"]
        self.batch_size = _TEST_MODEL["batch_size"]
        self.num_workers = _TEST_MODEL["num_workers"]

    def test_load_with_preprocess_labels(self):
        # Check column list and order.
        correct_columns = ["subject_id", "image_path"] + self.chexpert_label_list
        assert correct_columns == list(self.preprocessed_labels.columns)

        # Check data type of each dataset's column.
        correct_data_type = {
            "subject_id": "object", 
            "image_path": "object", 
        }
        for label in self.chexpert_label_list:
            correct_data_type[label] = "float"
        for column, dtype in correct_data_type.items():
            assert self.preprocessed_labels[column].dtype == dtype

        # Check can read image usually. If can't, occurr error by basic python system
        for image_path in self.preprocessed_labels["image_path"][:10]: # Too long to check all images
            with Image.open(image_path) as image:
                image.convert("RGB")

    def test_split_labels(self):
        labels_set = self.data_setter.split_labels(self.preprocessed_labels, self.seed)

        train_labels = labels_set["train_labels"]
        validation_labels = labels_set["validation_labels"]
        test_labels = labels_set["test_labels"]

        # Check data point numbers
        assert len(self.raw_labels) == (
            len(train_labels) + len(validation_labels) + len(test_labels)
        )

        # Check subject_id seperation
        all_ids = self.raw_labels["subject_id"].unique().tolist()

        train_ids = train_labels["subject_id"].unique().tolist()
        validation_ids = validation_labels["subject_id"].unique().tolist()
        test_ids = test_labels["subject_id"].unique().tolist()

        ## Check all ids seperated to each dataset without omission
        assert set(all_ids) == set(train_ids + validation_ids + test_ids)

        ## Check for duplicates of subject_id in two dataset
        assert set(train_ids).isdisjoint(validation_ids)
        assert set(train_ids).isdisjoint(test_ids)
        assert set(validation_ids).isdisjoint(test_ids)

        # Check dataset split ratio based on subject_id
        subject_id_numbers = len(all_ids)

        train_subject_id_numbers = len(train_ids)
        validation_subject_id_numbers = len(validation_ids)
        test_subject_id_numbers = len(test_ids)

        tolerance = 1e-2
        assert abs(train_subject_id_numbers / subject_id_numbers - 0.6) < tolerance
        assert abs(validation_subject_id_numbers / subject_id_numbers - 0.2) < tolerance
        assert abs(test_subject_id_numbers / subject_id_numbers - 0.2) < tolerance

    def test_get_data_loader(self):
        test_batch_size = self.batch_size
        test_labels = self.data_setter.split_labels(
            self.preprocessed_labels, self.seed
        )["test_labels"]
        test_dataloader = self.data_setter.get_data_loader(
            labels=test_labels, 
            chexpert_label_list=self.chexpert_label_list, 
            image_transform=self.image_test_transform, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
        )
        assert len(test_labels) == len(test_dataloader.dataset)

        images: torch.Tensor
        self.raw_labels: torch.Tensor
        images, self.raw_labels = next(iter(test_dataloader))
        assert images.shape == (test_batch_size, 3, 384, 384)
        assert self.raw_labels.shape == (test_batch_size, len(self.chexpert_label_list))
        assert self.raw_labels[0].dtype == torch.float32
