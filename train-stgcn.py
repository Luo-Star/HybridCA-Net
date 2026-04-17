import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from DataLoader_k_fold import FusionDataset
from net.st_gcn import Model as STGCN
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, patience=100, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, train_loss):
        if self.best_score is None or train_loss < self.best_score - self.delta:
            self.best_score = train_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(config):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{current_time}.log")

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info("Training configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")

    return log_filename


def train_model(model, train_loader, criterion, optimizer, num_epochs, log_filename, fold):
    early_stopping = EarlyStopping(patience=100)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for _, gcn_data, labels in train_loader:
            gcn_data, labels = gcn_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(gcn_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")

        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            logging.info(f"Fold {fold + 1}, Early stopping")
            break


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for _, gcn_data, labels in val_loader:
            gcn_data, labels = gcn_data.to(device), labels.to(device)
            outputs = model(gcn_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    val_loss /= len(val_loader)

    num_classes = len(set(all_targets))
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        roc_auc = roc_auc_score(all_targets, all_preds)
        return val_loss, accuracy, sensitivity, specificity, roc_auc
    else:
        report = classification_report(all_targets, all_preds, output_dict=True)
        sensitivities = {label: metrics['recall'] for label, metrics in report.items() if label.isdigit()}
        specificities = {}
        cm = confusion_matrix(all_targets, all_preds)
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            fn = cm[i, :].sum() - cm[i, i]
            tp = cm[i, i]
            specificities[str(i)] = tn / (tn + fp)

        all_targets_bin = label_binarize(all_targets, classes=list(range(num_classes)))
        all_preds_bin = label_binarize(all_preds, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(all_targets_bin, all_preds_bin, average="macro")
        sensitivity = sum(sensitivities.values()) / num_classes
        specificity = sum(specificities.values()) / num_classes
        return val_loss, accuracy, sensitivity, specificity, roc_auc


def get_kfold_dataloaders(config, num_folds=10):
    root_dir = config['root_dir']
    fmri_cut_size = config['fmri_cut_size']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    classes = config['classes']

    dataset = FusionDataset(root_dir, fmri_cut_size=fmri_cut_size, split='all', classes=classes)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_dataloaders = []
    for train_indices, val_indices in kf.split(dataset):
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        fold_dataloaders.append((train_loader, val_loader))

    return fold_dataloaders


if __name__ == '__main__':
    config_path = 'config.yaml'
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_filename = setup_logging(config)
    fold_dataloaders = get_kfold_dataloaders(config)

    num_epochs = config['num_epochs']
    class_names = config['classes']

    all_val_losses = []
    all_val_accuracies = []
    all_sensitivities = []
    all_specificities = []
    all_roc_aucs = []

    for fold, (train_loader, val_loader) in enumerate(fold_dataloaders):
        stgcn_model = STGCN(1, len(class_names), None, True).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(stgcn_model.parameters(), lr=config['learning_rate'])

        logging.info(f"Fold {fold + 1}/{len(fold_dataloaders)}")
        train_model(stgcn_model, train_loader, criterion, optimizer, num_epochs, log_filename, fold)
        val_loss, val_accuracy, sensitivity, specificity, roc_auc = validate_model(stgcn_model, val_loader, criterion)
        logging.info(f"Fold {fold + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        logging.info(
            f"Fold {fold + 1}: Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, ROC AUC: {roc_auc:.4f}")

        all_val_losses.append(val_loss)
        all_val_accuracies.append(val_accuracy)
        all_sensitivities.append(sensitivity)
        all_specificities.append(specificity)
        all_roc_aucs.append(roc_auc)

    avg_val_loss = np.mean(all_val_losses)
    avg_val_accuracy = np.mean(all_val_accuracies)
    avg_sensitivity = np.mean(all_sensitivities)
    avg_specificity = np.mean(all_specificities)
    avg_roc_auc = np.mean(all_roc_aucs)

    logging.info(f"Average Validation Loss: {avg_val_loss:.4f}")
    logging.info(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")
    logging.info(f"Average Sensitivity: {avg_sensitivity:.4f}")
    logging.info(f"Average Specificity: {avg_specificity:.4f}")
    logging.info(f"Average ROC AUC: {avg_roc_auc:.4f}")
