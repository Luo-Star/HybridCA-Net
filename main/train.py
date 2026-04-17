import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from DataLoader_k_fold import FusionDataset  # 确保路径正确
from net.ResNet3D import resnet3d18  # 确保路径正确
import logging
from datetime import datetime
from net.st_gcn import Model
from net.fusion_attension import FusionModel
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

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

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, features):
        loss = 0.0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                loss += F.l1_loss(features[i], features[j])
        return loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, lambda1, log_filename, fold, class_names_str, train_single_sample=False, use_consistency_loss=True):
    consistency_criterion = ConsistencyLoss()
    early_stopping = EarlyStopping(patience=100)
    


    best_model_wts = None  # 用于保存最佳模型的权重
    best_loss = float('inf')  # 初始化最佳损失值
    
    if train_single_sample:
        # 仅训练一个样本
        model.train()
        fmri_data, gcn_data, labels = next(iter(train_loader))
        fmri_data, gcn_data, labels = fmri_data.to(device), gcn_data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, resnet_output, gcn_output = model(fmri_data, gcn_data)
        classification_loss = criterion(outputs, labels)
        if use_consistency_loss:
            consistency_loss = consistency_criterion([resnet_output, gcn_output])
            loss = classification_loss + lambda1 * consistency_loss
        else:
            loss = classification_loss
        loss.backward()
        optimizer.step()

        logging.info(f"Fold {fold + 1} - Training Loss: {loss.item():.4f}")
    else:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for fmri_data, gcn_data, labels in train_loader:
                fmri_data, gcn_data, labels = fmri_data.to(device), gcn_data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, resnet_output, gcn_output = model(fmri_data, gcn_data)
                classification_loss = criterion(outputs, labels)
                if use_consistency_loss:
                    consistency_loss = consistency_criterion([resnet_output, gcn_output])
                    loss = classification_loss + lambda1 * consistency_loss
                else:
                    loss = classification_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")

            early_stopping(epoch_loss)
            if early_stopping.early_stop:
                logging.info(f"Fold {fold + 1}, Early stopping")
                break
           # 保存当前折的模型
        torch.save(model.state_dict(), f'model_fold_{fold+1}_{class_names_str}.pth')
        logging.info(f"Fold {fold + 1}, Training complete. Best model saved with lowest validation loss")


def validate_model(model, val_loader, criterion, lambda1, consistency_criterion, use_consistency_loss=True):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for fmri_data, gcn_data, labels in val_loader:
            fmri_data, gcn_data, labels = fmri_data.to(device), gcn_data.to(device), labels.to(device)
            outputs, resnet_output, gcn_output = model(fmri_data, gcn_data)
            classification_loss = criterion(outputs, labels)
            if use_consistency_loss:
                consistency_loss = consistency_criterion([resnet_output, gcn_output])
                loss = classification_loss + lambda1 * consistency_loss
            else:
                loss = classification_loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    val_loss /= len(val_loader)
    
        # Determine number of classes
    num_classes = len(set(all_targets))

    if num_classes == 2:
        # For binary classification
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        roc_auc = roc_auc_score(all_targets, all_preds)
        return val_loss, accuracy, sensitivity, specificity, roc_auc
    else:
        # For multiclass classification
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

        # Calculate macro average for sensitivities and specificities
        sensitivity = sum(sensitivities.values()) / num_classes
        specificity = sum(specificities.values()) / num_classes

        return val_loss, accuracy, sensitivity, specificity, roc_auc
            
    # Calculate sensitivity, specificity, ROC AUC
#     tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#     roc_auc = roc_auc_score(all_targets, all_preds)

#     return val_loss, accuracy, sensitivity, specificity, roc_auc


def test_model(model, test_loader, criterion, class_names):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for fmri_data, gcn_data, labels in test_loader:
            fmri_data, gcn_data, labels = fmri_data.to(device), gcn_data.to(device), labels.to(device)
            outputs, _, _ = model(fmri_data, gcn_data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    accuracy = 100 * correct / total
    logging.info(f"Test Loss: {test_loss / len(test_loader):.4f} - Test Accuracy: {accuracy:.2f}%")

    plot_confusion_matrix(all_labels, all_predictions, class_names)
    plot_roc_curve(all_labels, np.array(all_probs), class_names)

    return test_loss / len(test_loader), accuracy

def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage)')
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_roc_curve(y_true, y_score, classes, save_path='roc_curve.png'):
    y_true = label_binarize(y_true, classes=range(len(classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=600)
    plt.close()

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

    lambda1 = config.get('lambda1', 0.1)
    num_epochs = config['num_epochs']
    class_names = config['classes']
    class_names_str = '_'.join(class_names)
    attention_type = config['attention_type']
    use_consistency_loss = config['use_consistency_loss']

    all_val_losses = []
    all_val_accuracies = []
    all_sensitivities = []
    all_specificities = []
    all_roc_aucs = []

    for fold, (train_loader, val_loader) in enumerate(fold_dataloaders):
        # 模型重定义
        resnet3d_model = resnet3d18(num_classes=512, dropout_prob=0.5)
        stgcn_model = Model(1, len(config['classes']), None, True)
        fusion_model = FusionModel(resnet3d_model, stgcn_model, num_classes=len(config['classes']), attention_type=attention_type).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fusion_model.parameters(), lr=config['learning_rate'])

        logging.info(f"Fold {fold + 1}/{len(fold_dataloaders)}")
        train_model(fusion_model, train_loader, val_loader, criterion, optimizer, num_epochs, lambda1, log_filename, fold, class_names_str, False, use_consistency_loss=use_consistency_loss)
        val_loss, val_accuracy, sensitivity, specificity, roc_auc =  validate_model(fusion_model, val_loader, criterion, lambda1, ConsistencyLoss(), use_consistency_loss=use_consistency_loss)
        logging.info(f"Fold {fold + 1}: Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        logging.info(f"Fold {fold + 1}: Sensitivity: {sensitivity}, Specificity: {specificity}, ROC AUC: {roc_auc}")

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