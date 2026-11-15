import dataclasses
from tqdm import tqdm

from model.training_setter import CustomModel
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModelTrainerConfig:

    model: CustomModel 
    device: str 
    train_loader: torch.utils.data.DataLoader 
    validation_loader: torch.utils.data.DataLoader 
    test_loader: torch.utils.data.DataLoader 
    criterion: torch.nn.modules.loss._Loss 
    optimizer: torch.optim.Optimizer 
    warm_up_epochs: int 
    training_epochs: int 
    model_parameter_path: str 
    chexpert_label_list: list[str] 
    scheduler: torch.optim.lr_scheduler 


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig):
        model = model_trainer_config.model
        device = model_trainer_config.device
        model_parameter_path = model_trainer_config.model_parameter_path
        chexpert_label_list = model_trainer_config.chexpert_label_list

        self.warm_up_model_config = {
            "model": model, 
            "device": device, 
            "train_loader": model_trainer_config.train_loader, 
            "criterion": model_trainer_config.criterion, 
            "optimizer": model_trainer_config.optimizer, 
            "epochs": model_trainer_config.warm_up_epochs, 
            "model_parameter_path": model_parameter_path, 
        }
        self.train_model_config = {
            "model": model, 
            "device": device, 
            "train_loader": model_trainer_config.train_loader, 
            "validation_loader": model_trainer_config.validation_loader, 
            "criterion": model_trainer_config.criterion, 
            "optimizer": model_trainer_config.optimizer, 
            "epochs": model_trainer_config.training_epochs, 
            "model_parameter_path": model_parameter_path, 
            "chexpert_label_list": chexpert_label_list, 
            "scheduler": model_trainer_config.scheduler, 
        }
        self.evaluate_model_config = {
            "model": model,
            "device": device,
            "test_loader": model_trainer_config.test_loader,
            "criterion": model_trainer_config.criterion,
            "model_parameter_path": model_parameter_path,
            "chexpert_label_list": chexpert_label_list,
        }

    def warm_up_model(
        self, 
        model: CustomModel, 
        device: str, 
        train_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer, 
        epochs: int, 
        model_parameter_path: str, 
    ):
        print(f"\n# Warm-up model ({epochs} epochs)")
        model.set_grad(features_freeze=True)
        for epoch in range(epochs):
            _ = self._train_per_epoch(
                model, device, train_loader, criterion, optimizer, epoch
            )
        torch.save(model.state_dict(), model_parameter_path)

    def train_model(
        self, 
        model: CustomModel, 
        device: str, 
        train_loader: torch.utils.data.DataLoader, 
        validation_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer, 
        epochs: int, 
        model_parameter_path: str, 
        chexpert_label_list: list[str], 
        scheduler: torch.optim.lr_scheduler, 
    ):
        print(f"\n# Train model ({epochs} epochs)")
        model.set_grad(features_freeze=False)
        best_auroc = float("-inf")
        for epoch in range(epochs):
            train_loss = self._train_per_epoch(
                model, device, train_loader, criterion, optimizer, epoch
            )
            validation_loss, class_aurocs, class_auprcs = self._evaluate_per_epoch(
                model, device, validation_loader, criterion, result="none"
            )

            scheduler.step()
            auroc = np.mean(class_aurocs) if len(class_aurocs) > 0 else np.nan
            auprc = np.mean(class_auprcs) if len(class_auprcs) > 0 else np.nan

            print(f"\n# Epoch {epoch + 1}: Evaluate model use Validate dataset")
            print(f"| train loss: {train_loss:.4f} | valid loss: {validation_loss:.4f}")
            print(f"| auroc: {auroc:.4f} | auprc: {auprc:.4f} |")
            print(f"# AUROC")
            for i, label in enumerate(chexpert_label_list):
                print(f"| {label}: {class_aurocs[i]:.4f}", end=" ")
                if (i % 4 == 0) and (i != 0): 
                    print("|")

            if auroc > best_auroc:
                torch.save(model.state_dict(), model_parameter_path)
                print(f"# Model update: {best_auroc:.4f} → {auroc:.4f}")
                best_auroc = auroc

    def _train_per_epoch(
        self,
        model: CustomModel, 
        device: str, 
        dataloader: torch.utils.data.DataLoader, 
        criterion: nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer, 
        epoch: int,
    ) -> float:
        model.to(device)
        model.train()

        total_loss, total_valid = 0.0, 0
        dataloader_tqdm = tqdm(
            dataloader, 
            desc=f'Training Epoch {epoch + 1}', 
            total=len(dataloader), 
            leave=False,
        )
        for images, labels in dataloader_tqdm:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) # (batch size, n_class)

            valid_mask = (~torch.isnan(labels)) & (labels != -1.0)

            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue  # 이 batch는 완전히 skip

            labels_clean = labels.clone()
            labels_clean[torch.isnan(labels_clean)] = 0.0
            labels_clean[labels_clean == -1.0] = 0.0

            optimizer.zero_grad()

            logits = model(images) # (batch size, n_class)

            loss_elem = criterion(logits, labels_clean)
            loss_masked = loss_elem * valid_mask.float()
            loss = loss_masked.sum() / valid_count

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * valid_count
            total_valid += valid_count
        dataloader_tqdm.close()

        if total_valid == 0:
            return 0.0
        return total_loss / total_valid

    def evaluate_model(
        self,
        model: CustomModel, 
        device: str, 
        test_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.modules.loss._Loss, 
        model_parameter_path: str, 
        chexpert_label_list: list[str], 
    ):
        print(f"\n# Evaludate model use Test dataset")
        model.load_state_dict(torch.load(model_parameter_path))
        model.to(device)

        _, class_aurocs, class_auprcs = self._evaluate_per_epoch(
            model, device, test_loader, criterion, result="none"
        ) 
        auroc = np.mean(class_aurocs) if len(class_aurocs) > 0 else np.nan
        auprc = np.mean(class_auprcs) if len(class_auprcs) > 0 else np.nan

        print(f"| auroc: {auroc:.4f} | auprc: {auprc:.4f} |")
        print(f"# AUROC")
        for i, label in enumerate(chexpert_label_list):
            print(f"| {label}: {class_aurocs[i]:.4f}", end=" ")
            if (i % 4 == 0) and (i != 0):
                print("|")

    def _evaluate_per_epoch(
        self,
        model: CustomModel, 
        device: str, 
        dataloader: torch.utils.data.DataLoader, 
        criterion: torch.nn.modules.loss._Loss, 
        result: str = "mean",
    ) -> tuple[float, ...]:
        model.eval()

        probs_record, labels_record = [], []
        total_loss, total_valid = 0.0, 0
        dataloader_tqdm = tqdm(
            dataloader, desc="Validate/Test", total=len(dataloader), leave=False
        )
        with torch.no_grad():
            for images, labels in dataloader_tqdm:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True) 

                valid_mask = (~torch.isnan(labels)) & (labels != -1.0)
        
                valid_count = valid_mask.sum().item()
                if valid_count == 0: # 확실한 정보가 있는지 확인
                    continue
        
                labels_clean = labels.clone()
                labels_clean[torch.isnan(labels_clean)] = 0.0
                labels_clean[labels_clean == -1.0] = 0.0

                logits = model(images) # (batch size, n_class)

                loss_elem = criterion(logits, labels_clean)
                loss_masked = loss_elem * valid_mask.float()
                loss = loss_masked.sum() / valid_count
        
                total_loss += loss.item() * valid_count
                total_valid += valid_count

                probs = torch.sigmoid(logits)

                probs_record.append(probs.cpu())
                labels_record.append(labels.cpu())
        dataloader_tqdm.close()

        all_probs = torch.cat(probs_record, dim=0).numpy()
        all_labels = torch.cat(labels_record, dim=0).numpy()

        n_classes = all_labels.shape[1]
        class_aurocs, class_auprcs = [], []

        for i in range(n_classes):
            class_labels = all_labels[:, i]
            class_probs = all_probs[:, i]

            valid_mask = (class_labels != -1) & (~np.isnan(class_labels))
            if np.sum(valid_mask) == 0:
                continue

            y_true = class_labels[valid_mask]
            y_prob = class_probs[valid_mask]

            if len(np.unique(y_true)) > 1:
                try:
                    class_aurocs.append(roc_auc_score(y_true, y_prob))
                except Exception as e:
                    print(f"Warning: Class {i} AUROC calculation error: {e}")
            
            if np.sum(y_true == 1) > 0:
                try:
                    class_auprcs.append(average_precision_score(y_true, y_prob))
                except Exception as e:
                    print(f"Warning: Class {i} AUPRC calculation error: {e}")

        result = result.lower()
        average_loss = total_loss / total_valid
        if result == "mean":
            auroc_macro = np.mean(class_aurocs) if len(class_aurocs) > 0 else np.nan
            auprc_macro = np.mean(class_auprcs) if len(class_auprcs) > 0 else np.nan
            return average_loss, auroc_macro, auprc_macro
        elif result == "none":
            return average_loss, class_aurocs, class_auprcs

    def work(self):
        self.warm_up_model(**self.warm_up_model_config)
        self.train_model(**self.train_model_config)
        self.evaluate_model(**self.evaluate_model_config)
