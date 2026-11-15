from unittest.mock import patch

from config import Config
from data.data_setter import DataSetter
from model.models.convnext_base import ModelBasedConvNeXtBase
from model.training_setter import CustomModel
import pytest
import torch
from training_setter import TrainingSetter


_CONFIG = Config()
_ENVIRONMENT = _CONFIG.environment
_PATH = _CONFIG.path
_LABEL = _CONFIG.label
_MODEL = _CONFIG.model

TEST_MODEL_NAME = "convnext_base"
_TEST_MODEL = _MODEL[TEST_MODEL_NAME]


class TestTrainingSetter:

    @pytest.fixture(autouse=True)
    def setUp(self):
        data_setter = DataSetter(data_setter_config=None)
        labels = data_setter.load_raw_labels(str(_PATH["chexpert_labels_path"]))
        labels = data_setter.preprocess_labels(
            labels, str(_PATH["images"]), _LABEL["chexpert_label_list"]
        )
        train_labels = data_setter.split_labels(
            labels, _ENVIRONMENT["seed"]
        )["train_labels"]

        self.training_setter = TrainingSetter(training_setter_config=None)
        self.model_name = TEST_MODEL_NAME # if modify, also must change `test_select_model`
        self.model = self.training_setter.select_model(self.model_name, 13)
        self.device = self.training_setter.get_device()
        self.weight_decay = _TEST_MODEL["weight_decay"]
        self.warm_up = _TEST_MODEL["warm_up"]
        self.oprimizer = self.training_setter.get_optimizer(
            self.model, self.weight_decay, self.warm_up
        )
        self.criterion = self.training_setter.get_criterion(
            train_labels, self.device, _LABEL["chexpert_label_list"]
        )

    def test_select_model(self):
        model: CustomModel
        model = self.training_setter.select_model(self.model_name, 13)
        assert isinstance(model, ModelBasedConvNeXtBase)

        model_name = "temporary model name"
        with pytest.raises(ValueError, match=f"Do not support model: {model_name}"):
            self.training_setter.select_model(model_name, 13)

    def test_get_device(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert self.training_setter.get_device() == "cuda"

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                assert self.training_setter.get_device() == "mps"

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                assert self.training_setter.get_device() == "cpu"

    def test_get_optimizer(self):
        optimizer = self.training_setter.get_optimizer(
            self.model, self.weight_decay, self.warm_up
        )
        assert isinstance(optimizer, torch.optim.AdamW)

        param_groups = optimizer.param_groups
        assert param_groups[0]["weight_decay"] == self.weight_decay

        tolerance = 1e-10
        if self.warm_up:
            assert len(param_groups) == 1
            assert abs(param_groups[0]["lr"] - 1e-3) < tolerance
        elif not self.warm_up:
            assert len(param_groups) == 2
            assert abs(param_groups[1]["lr"] - 1e-5) < tolerance
        else:
            raise ValueError(f"Invalid warm_up value: {self.warm_up}")

    def test_get_criterion(self):
        data_setter = DataSetter(data_setter_config=None)
        labels = data_setter.load_raw_labels(str(_PATH["chexpert_labels_path"]))
        labels = data_setter.preprocess_labels(
            labels, str(_PATH["images"]), _LABEL["chexpert_label_list"]
        )
        train_labels = data_setter.split_labels(
            labels, _ENVIRONMENT["seed"]
        )["train_labels"]

        criterion = self.training_setter.get_criterion(
            train_labels, self.device, _LABEL["chexpert_label_list"]
        )
        assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)
        assert torch.equal(
            criterion.pos_weight, 
            self.training_setter._set_pos_weights(
                train_labels, _LABEL["chexpert_label_list"]
            ).to(self.device)
        )

    def test_get_scheduler(self):
        epochs = _TEST_MODEL["training_epochs"]
        minimum_scheduler_learning_rate = _TEST_MODEL["minimum_scheduler_learning_rate"]
        scheduler = self.training_setter.get_scheduler(
            self.oprimizer, 
            epochs,
            minimum_scheduler_learning_rate,
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

        tolerance = 1e-10
        assert scheduler.T_max == epochs
        assert abs(scheduler.eta_min - minimum_scheduler_learning_rate) < tolerance
