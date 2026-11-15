from config import Config
from data.data_setter import DataSetter
from data.data_setter import DataSetterConfig
from model.model_trainer import ModelTrainer
from model.model_trainer import ModelTrainerConfig
from model.training_setter import TrainingSetter
from model.training_setter import TrainingSetterConfig


_CONFIG = Config()
_ENVIRONMENT = _CONFIG.environment
_PATH = _CONFIG.path
_LABEL = _CONFIG.label
_IMAGE = _CONFIG.image
_MODEL = _CONFIG.model

TEST_MODEL_NAME = "convnext_base"
_TEST_MODEL = _MODEL[TEST_MODEL_NAME]


def main():
    seed = _ENVIRONMENT["seed"]
    images_dir = str(_PATH["images"])
    chexpert_labels_path = str(_PATH["chexpert_labels_path"])
    chexpert_label_list = _LABEL["chexpert_label_list"]
    image_transform_set = _IMAGE
    model_parameter_path = str(
        _MODEL["common"]["model_parameter_dir"] 
        / _TEST_MODEL["model_parameter_file_name"]
    )
    warm_up = _TEST_MODEL["warm_up"]
    warm_up_epochs = _TEST_MODEL["warm_up_epochs"]
    training_epochs = _TEST_MODEL["training_epochs"]
    batch_size = _TEST_MODEL["batch_size"]
    weight_decay = _TEST_MODEL["weight_decay"]
    minimum_scheduler_learning_rate = _TEST_MODEL["minimum_scheduler_learning_rate"]
    num_workers = _TEST_MODEL["num_workers"]

    data_setter = DataSetter(
        DataSetterConfig(
            images_dir=images_dir, 
            chexpert_label_list=chexpert_label_list, 
            chexpert_labels_path=chexpert_labels_path, 
            seed=seed, 
            image_transform_set=image_transform_set, 
            batch_size=batch_size, 
            num_workers=num_workers, 
        )
    )
    data_setter_outputs = data_setter.work()
    train_loader = data_setter_outputs["train_loader"]
    validation_loader = data_setter_outputs["validation_loader"]
    test_loader = data_setter_outputs["test_loader"]

    training_setter = TrainingSetter(
        TrainingSetterConfig(
            model_name=TEST_MODEL_NAME, 
            weight_decay=weight_decay, 
            warm_up=warm_up, 
            train_labels=train_loader.dataset, 
            chexpert_label_list=chexpert_label_list, 
            epochs=training_epochs, 
            minimum_scheduler_learning_rate=minimum_scheduler_learning_rate, 
        )
    )
    training_setter_outputs = training_setter.work()
    model = training_setter_outputs["model"]
    optimizer = training_setter_outputs["optimizer"]
    device = training_setter_outputs["device"]
    criterion = training_setter_outputs["criterion"]
    scheduler = training_setter_outputs["scheduler"]

    model_trainer = ModelTrainer(
        ModelTrainerConfig(
            model=model, 
            device=device, 
            train_loader=train_loader, 
            validation_loader=validation_loader, 
            test_loader=test_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            warm_up_epochs=warm_up_epochs, 
            training_epochs=training_epochs, 
            model_parameter_path=model_parameter_path, 
            chexpert_label_list=chexpert_label_list, 
            scheduler=scheduler, 
        )
    )
    model_trainer.work()
