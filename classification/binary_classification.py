"""
Project Name: cnn_transformer_artifact_classifier
Author: Aonan He
Contact: ahee0015@student.monash.edu
Last Updated: 2025-04-28
Copyright (c) 2025 Aonan He
Licensed under the MIT License
Description:
This script trains a hybrid CNN-Transformer model to perform fragment-level
EEG artifact binary classification. It loads preprocessed EEG segments,
performs hyperparameter optimization with Hyperopt, and saves the best model.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import warnings
import GPUtil
import logging
import sys

# -----------------------------------------------------------
# Configurable parameters 
# -----------------------------------------------------------
log_filename       = " "               # Training log file
data_dir           = " "   # Directory containing .pkl EEG segments
top2_labels        = ["close_base", "artifact"]            # Label names for classification
max_evals          = 80                                     # Max Hyperopt evaluations
# -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()
warnings.filterwarnings("ignore")

# Experiment configuration header
logger.info("########################")
logger.info("model_1: Fragment-level artifact binary classification")
logger.info("Input shape: 34x375, Hyperopt hyperparameter tuning")
logger.info("########################")

def select_free_gpu():
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        logger.info("No GPUs available, using CPU")
        return torch.device("cpu")
    best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    logger.info(f"Selected GPU: {best_gpu.id} - {best_gpu.name}, Free Memory: {best_gpu.memoryFree} MB")
    return torch.device(f"cuda:{best_gpu.id}")

device = select_free_gpu() # Initialize computing device

########################
# 1. Dataset and Model Definition
########################
class EEGArtifactDataset(Dataset):
    """Custom dataset for EEG artifact classification
    
    Attributes:
        label_to_idx (dict): Mapping from class names to indices
        samples (list): Filtered EEG samples with valid artifacts
    """
    def __init__(self, samples, label_to_idx):
        super().__init__()
        self.label_to_idx = label_to_idx

        # Filter samples: exclude 'close_base' and invalid entries
        filtered = []
        for s in samples:
            # Check required fields and non-empty artifact types
            if "eeg_data" in s and "artifact_types" in s:
                if s["artifact_types"] == ["close_base"]:
                    s["label"] = "close_base"
                else:
                    s["label"] = "artifact"
                if s["label"] in label_to_idx:
                    filtered.append(s)
        self.samples = filtered

        # Dataset statistics
        logger.info(f"Filtered dataset size: {len(self.samples)}")
        label_counter = Counter([s["label"] for s in self.samples])
        logger.info(f"Label distribution: {label_counter}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Convert raw EEG data to normalized tensors
        Args:
            idx (int): Sample index
        Returns:
            tuple: (eeg_tensor, label_index)
        """
        s = self.samples[idx]
        data_array = s["eeg_data"]
        scale_factor = 1e6 # Convert to microvolts (μV) scale
        data_array = data_array * scale_factor

        lbl_str = s["label"]
        lbl_idx = self.label_to_idx[lbl_str]
        data_tensor = torch.tensor(data_array, dtype=torch.float32)
        label_tensor = torch.tensor(lbl_idx, dtype=torch.long)
        return data_tensor, label_tensor

class CNNTransformerModel(nn.Module):
    """Hybrid CNN-Transformer architecture for temporal-spatial EEG analysis
    Architecture:
        1. 2D CNN for local feature extraction
        2. Transformer encoder for temporal dependencies
        3. Fully-connected layers for classification
    Args:
        num_classes (int): Number of output classes
        cnn_filters (list): Convolutional filter sizes per block
        transformer_nhead (int): Number of attention heads
        transformer_ff (int): Feed-forward dimension in transformer
        transformer_layers (int): Number of transformer encoder layers
    """
    def __init__(
        self,
        num_classes=2,        
        cnn_filters=[8, 16, 32, 64, 128],  
        transformer_nhead=8,   
        transformer_ff=1024,   
        transformer_layers=1  
    ):
        super().__init__()

        # CNN Module: Process EEG as 2D image (1 channel x 34 ch x 375 timepoints)
        convs = []
        in_ch = 1  
        for out_ch in cnn_filters:
            convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), padding=(1,1)),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                )
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*convs)

        # Transformer Configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_filters[-1],      
            nhead=transformer_nhead,      
            dim_feedforward=transformer_ff,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        fc_in_dim = cnn_filters[-1]
        self.fc1 = nn.Linear(fc_in_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(100, num_classes)

    def forward(self, x_data):
        """Forward pass through network
        Args:
            x_data (Tensor): Input EEG data (batch x 34 x 375)
        Returns:
            Tensor: Class logits (batch x num_classes)
        """
        x_data = x_data.unsqueeze(1)  
        x_cnn = self.cnn(x_data)      
        x_cnn = x_cnn.flatten(2)      
        x_cnn = x_cnn.transpose(1, 2) #  (batch, sequence_length, out_channels)
        x_trf = self.transformer_encoder(x_cnn)
        x_trf = x_trf.transpose(1, 2) # (batch, out_channels, sequence_length)
        pooled = self.global_pool(x_trf).squeeze(-1)  # (batch, out_channels)
        x = self.fc1(pooled)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        logits = self.out(x)
        return logits

########################
# 2. Training Function
########################
def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10, patience=5):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}") as pbar:
            for data_tensor, label_tensor in pbar:
                data_tensor, label_tensor = data_tensor.to(device), label_tensor.to(device)
                # Forward pass
                optimizer.zero_grad()
                logits = model(data_tensor)
                loss = criterion(logits, label_tensor)
                # Backpropagation
                loss.backward()
                optimizer.step()

                # Metrics tracking
                train_loss += loss.item() * data_tensor.size(0)
                train_correct += (logits.argmax(1) == label_tensor).sum().item()
                pbar.set_postfix(loss=loss.item())

        # Validation Phase
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_preds = []
        val_labels = []
        with tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}") as pbar:
            with torch.no_grad():
                for data_tensor, label_tensor in pbar:
                    data_tensor, label_tensor = data_tensor.to(device), label_tensor.to(device)
                    logits = model(data_tensor)
                    loss = criterion(logits, label_tensor)

                    # Accumulate validation metrics
                    val_loss += loss.item() * data_tensor.size(0)
                    val_correct += (logits.argmax(1) == label_tensor).sum().item()

                    # Store predictions for final evaluation
                    val_preds.extend(logits.argmax(1).cpu().numpy())
                    val_labels.extend(label_tensor.cpu().numpy())
                    pbar.set_postfix(loss=loss.item())

         # Epoch Statistics
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "new_best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered!")
                break

    # Final Evaluation Metrics
    logger.info("\nClassification Report:")
    report = classification_report(val_labels, val_preds, target_names=top2_labels)
    logger.info(report)
    cm = confusion_matrix(val_labels, val_preds)
    logger.info("\nConfusion Matrix:")
    for i, row in enumerate(cm):
        logger.info(f"Class {top2_labels[i]}: {row}")

    return best_val_loss, val_acc

########################
# 3. Main Program
########################

if __name__ == "__main__":
    samples = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".pkl"):
            filepath = os.path.join(data_dir, fname)
            with open(filepath, 'rb') as f:
                arr = pickle.load(f)
            for sample_dict in arr:
                if "eeg_data" in sample_dict and "artifact_types" in sample_dict:
                    samples.append(sample_dict)

    logger.info(f"Total samples loaded: {len(samples)}")

    # Dataset Preparation
    label_to_idx = {lbl: i for i, lbl in enumerate(top2_labels)}
    dataset = EEGArtifactDataset(samples, label_to_idx)
    test_split_ratio = 0.2
    test_size = int(len(dataset) * test_split_ratio)
    train_size = len(dataset) - test_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    # Class Weight Calculation (for imbalanced data)
    label_counts = Counter([s["label"] for s in samples if "label" in s])
    total_samples = sum(label_counts.values())
    initial_class_weights = [total_samples / label_counts[label] for label in label_to_idx.keys()]

    logger.info("\nInitial Class Weights:")
    for label, weight in zip(label_to_idx.keys(), initial_class_weights):
        logger.info(f"Class '{label}' - Weight: {weight:.4f}")

    # Hyperparameter Search Space (for Hyperopt)
    space = {
        'lr': hp.loguniform('lr', np.log(1e-10), np.log(1e-3)),
        'batch_size': hp.choice('batch_size', [128, 256]),
        'transformer_layers': hp.choice('transformer_layers', [1, 2, 3]),
        'transformer_nhead': hp.choice('transformer_nhead', [2, 4, 8, 16]),
        'num_conv_blocks': hp.choice('num_conv_blocks', [3, 4, 5]),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
        'class_weight_scaling_0': hp.uniform('class_weight_scaling_0', 0.5, 2.0),
        'class_weight_scaling_1': hp.uniform('class_weight_scaling_1', 0.5, 2.0)
    }

    def objective(params):
        lr = params['lr']
        batch_size = params['batch_size']
        transformer_layers = params['transformer_layers']
        transformer_nhead = params['transformer_nhead']
        num_conv_blocks = params['num_conv_blocks']
        weight_decay = params['weight_decay']
        class_weight_scaling = [
            params['class_weight_scaling_0'],
            params['class_weight_scaling_1']
        ]

        logger.info(f"\nTesting with params: {params}")
        # Adjust class weights with scaling factors
        adjusted_class_weights = [initial_weight * scaling for initial_weight, scaling in zip(initial_class_weights, class_weight_scaling)]
        class_weights_tensor = torch.tensor(adjusted_class_weights, dtype=torch.float32).to(device)

        logger.info("Adjusted Class Weights:")
        for label, weight in zip(label_to_idx.keys(), adjusted_class_weights):
            logger.info(f"Class '{label}' - Weight: {weight:.4f}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        base_filters = [8, 16, 32, 64, 128]
        cnn_filters = base_filters[:num_conv_blocks]

        # Model Initialization with current hyperparameters
        model = CNNTransformerModel(
            num_classes=len(label_to_idx),
            cnn_filters=cnn_filters,
            transformer_layers=transformer_layers,
            transformer_nhead=transformer_nhead
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_epochs = 15
        patience = 3

        val_loss, val_acc = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            patience=patience
        )

        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        # Clear GPU cache between trials
        torch.cuda.empty_cache()
        return {'loss': val_loss, 'status': STATUS_OK}

    # Hyperparameter Optimization with TPE algorithm
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals= max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Final Model Training with Best Parameters
    logger.info("\nBest hyperparameters:")
    logger.info(best)
    best_params = {
        'lr': best['lr'],
        'batch_size': [128, 256][best['batch_size']],
        'transformer_layers': [1, 2, 3][best['transformer_layers']],
        'transformer_nhead': [2, 4, 8, 16][best['transformer_nhead']],
        'num_conv_blocks': [3, 4, 5][best['num_conv_blocks']],
        'weight_decay': best['weight_decay'],
        'class_weight_scaling': [
            best['class_weight_scaling_0'],
            best['class_weight_scaling_1']
        ]
    }
    logger.info("Best hyperparameters (mapped):")
    logger.info(best_params)

    logger.info("\nStarting final training with best hyperparameters...")
    final_class_weights = [initial_weight * scaling for initial_weight, scaling in zip(initial_class_weights, best_params['class_weight_scaling'])]
    final_class_weights_tensor = torch.tensor(final_class_weights, dtype=torch.float32).to(device)

    logger.info("Final Adjusted Class Weights:")
    for label, weight in zip(label_to_idx.keys(), final_class_weights):
        logger.info(f"Class '{label}' - Weight: {weight:.4f}")

    final_train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    final_val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    base_filters = [8, 16, 32, 64, 128]
    cnn_filters = base_filters[:best_params['num_conv_blocks']]
    final_model = CNNTransformerModel(
        num_classes=len(label_to_idx),
        cnn_filters=cnn_filters,
        transformer_layers=best_params['transformer_layers'],
        transformer_nhead=best_params['transformer_nhead']
    ).to(device)

    final_criterion = nn.CrossEntropyLoss(weight=final_class_weights_tensor)
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    final_num_epochs = 15
    final_patience = 3

    best_val_loss, best_val_acc = train_model(
        model=final_model,
        criterion=final_criterion,
        optimizer=final_optimizer,
        train_loader=final_train_loader,
        val_loader=final_val_loader,
        device=device,
        num_epochs=final_num_epochs,
        patience=final_patience
    )

    logger.info(f"\nFinal Model - Best Validation Loss: {best_val_loss:.4f}, Best Validation Acc: {best_val_acc:.4f}")
    torch.save(final_model.state_dict(), "new_cnn_transformer_artifact_classifier.pth")
    logger.info("Final model saved to 'new_cnn_transformer_artifact_classifier.pth'")
    torch.cuda.empty_cache()