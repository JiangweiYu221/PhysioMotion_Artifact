#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-class EEG artifact classification with nested subject-level GroupKFold and Hyperopt.

What this script does
- Loads windowed EEG samples from per-subject .pkl files (subX_runY.pkl).
- Performs nested cross-validation with GroupKFold:
  - Outer folds: subject-independent evaluation on the held-out test subjects.
  - Inner folds: hyperparameter tuning (Hyperopt/TPE) on train/val subjects.
- Trains a CNN + Transformer encoder baseline.
- Logs detailed metrics and confusion matrices WITHOUT changing the existing log keywords
  (so your downstream parsing scripts can keep working).

Important assumptions about your data format
- Each .pkl file contains a list of dict items.
- Each item has:
  - "eeg_data": array-like shaped (C, T) (e.g., 34 x 375)
  - "artifact_types": list[str], where the first entry is treated as the label
  - "close_base" samples are discarded for multi-class training.

Reproducibility
- Random seeds are fixed (NumPy/PyTorch/python).
"""

import os
import re
import sys
import math
import time
import pickle
import random
import logging
import warnings
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

try:
    import GPUtil
except Exception:
    GPUtil = None

warnings.filterwarnings("ignore")


# -------------------------
# Configuration
# -------------------------

@dataclass
class Config:
    # Data
    data_dir: str = "/home/mnt_disk1/Motion_Artifact_han1/Slidingwindow"
    artifact_labels: Tuple[str, ...] = (
        "blink", "chew", "eyebrow", "blink_hor_headm", "hor_headm",
        "tongue", "tongue_eyebrow", "ver_headm", "blink_eyebrow",
        "blink_ver_headm", "hor_eyem", "ver_eyem", "swallow", "swallow_eyebrow"
    )

    # Nested CV
    outer_k: int = 5
    inner_k: int = 3
    max_evals: int = 30
    seed: int = 42

    # Training
    inner_num_epochs: int = 10
    inner_patience: int = 2
    final_num_epochs: int = 20
    final_patience: int = 5

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True

    # Logging (do not change keywords to keep parsers stable)
    log_filename: str = "model2_nested_groupkfold_verbose_log.txt"
    log_every_epoch: bool = True
    log_inner_fold_cm: bool = False
    log_trial_cm_agg: bool = True
    log_trial_params: bool = True
    log_outer_test_cm: bool = True

    # Hyperopt space: same as your current design
    batch_size_choices: Tuple[int, ...] = (128, 256)
    transformer_layers_choices: Tuple[int, ...] = (1, 2, 3)
    transformer_nhead_choices: Tuple[int, ...] = (2, 4, 8, 16)
    num_conv_blocks_choices: Tuple[int, ...] = (4, 5)  # you intentionally dropped 3

    # Weight decay bucket priors + ranges
    wd_bucket_probs: Tuple[float, ...] = (0.25, 0.40, 0.25, 0.10)  # tiny/small/mid/large
    wd_tiny_range: Tuple[float, float] = (1e-7, 3e-6)
    wd_small_range: Tuple[float, float] = (3e-6, 3e-5)
    wd_mid_range: Tuple[float, float] = (3e-5, 3e-4)
    wd_large_range: Tuple[float, float] = (3e-4, 3e-3)

    # Structured class-weight scaling: global * (1 + delta_i), then clipped
    cw_global_range: Tuple[float, float] = (0.85, 1.35)
    cw_delta_range: Tuple[float, float] = (-0.25, 0.25)
    cw_clip: Tuple[float, float] = (0.5, 2.0)

    # Misc
    eeg_scale: float = 1e6  # scale to microvolts


CFG = Config()
NUM_CLASSES = len(CFG.artifact_labels)


# -------------------------
# Logging
# -------------------------

def setup_logger(log_filename: str) -> logging.Logger:
    logger = logging.getLogger("model2_nested_cv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_filename, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = setup_logger(CFG.log_filename)


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# GPU selection
# -------------------------

def select_free_gpu() -> torch.device:
    """Select the GPU with most free memory; fall back to CPU if unavailable."""
    if (GPUtil is None) or (not torch.cuda.is_available()):
        logger.info("GPUtil not available or CUDA not available, using CPU.")
        return torch.device("cpu")

    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        logger.info("No GPUs available, using CPU.")
        return torch.device("cpu")

    best = max(gpus, key=lambda g: g.memoryFree)
    logger.info(f"Selected GPU: {best.id} - {best.name}, Free Memory: {best.memoryFree} MB")
    return torch.device(f"cuda:{best.id}")


# -------------------------
# Data loading
# -------------------------

SUBJECT_RE = re.compile(r"(sub\d+)_run\d+\.pkl$", re.IGNORECASE)


def parse_subject_id(filename: str) -> str:
    m = SUBJECT_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse subject_id from filename: {filename}. Expected like sub1_run01.pkl")
    return m.group(1).lower()


def load_all_samples_with_subject(data_dir: str) -> List[dict]:
    """Load all samples from .pkl files and attach subject_id parsed from file name."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    samples: List[dict] = []
    for fname in pkl_files:
        subject_id = parse_subject_id(fname)
        path = os.path.join(data_dir, fname)
        with open(path, "rb") as f:
            arr = pickle.load(f)

        for d in arr:
            if isinstance(d, dict) and ("eeg_data" in d) and ("artifact_types" in d):
                dd = dict(d)
                dd["subject_id"] = subject_id
                samples.append(dd)

    return samples


# -------------------------
# Dataset (multi-class; drop close_base)
# -------------------------

class EEGArtifactDatasetMulti(Dataset):
    """
    Multi-class dataset.
    - Drop samples whose first artifact label is 'close_base'.
    - Use the first entry of 'artifact_types' as the class label.
    """

    def __init__(self, samples: List[dict], label_to_idx: Dict[str, int], eeg_scale: float = 1e6):
        super().__init__()
        self.label_to_idx = label_to_idx
        self.eeg_scale = float(eeg_scale)

        filtered = []
        for s in samples:
            if ("eeg_data" not in s) or ("artifact_types" not in s) or ("subject_id" not in s):
                continue
            if not s["artifact_types"]:
                continue
            if s["artifact_types"][0] == "close_base":
                continue

            label = s["artifact_types"][0]
            if label in label_to_idx:
                ss = dict(s)
                ss["label"] = label
                filtered.append(ss)

        self.samples = filtered

        logger.info(f"Filtered dataset size: {len(self.samples)}")
        label_counter = Counter([s["label"] for s in self.samples])
        subj_counter = Counter([s["subject_id"] for s in self.samples])
        logger.info(f"Label distribution: {label_counter}")
        logger.info(
            f"Num subjects: {len(subj_counter)} | subject sample range: "
            f"min={min(subj_counter.values())}, max={max(subj_counter.values())}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = np.asarray(s["eeg_data"], dtype=np.float32) * self.eeg_scale
        y = self.label_to_idx[s["label"]]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# -------------------------
# Model: CNN + Transformer
# -------------------------

class CNNTransformerModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_filters: List[int] = None,
        transformer_nhead: int = 8,
        transformer_ff: int = 1024,
        transformer_layers: int = 1,
    ):
        super().__init__()
        if cnn_filters is None:
            cnn_filters = [8, 16, 32, 64, 128]

        convs = []
        in_ch = 1
        for out_ch in cnn_filters:
            convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*convs)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_filters[-1],
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(cnn_filters[-1], 100)
        self.fc2 = nn.Linear(100, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)           # (B,1,C,T)
        x = self.cnn(x)              # (B,Ch,H,W)
        x = x.flatten(2).transpose(1, 2)  # (B,seq,Ch)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)        # (B,Ch,seq)
        x = self.global_pool(x).squeeze(-1)  # (B,Ch)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


# -------------------------
# Train / eval helpers
# -------------------------

def compute_base_class_weights_maxnorm(dataset: EEGArtifactDatasetMulti, indices: np.ndarray, label_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Compute class weights using max-normalized inverse frequency:
      w_k = max(counts) / counts_k
    """
    counts = np.zeros(len(label_to_idx), dtype=np.float64)
    for idx in indices:
        lbl = dataset.samples[int(idx)]["label"]
        counts[label_to_idx[lbl]] += 1.0
    counts = np.maximum(counts, 1.0)
    max_count = float(counts.max())
    weights = max_count / counts
    return weights.astype(np.float32)


def predict_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) on the given loader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    return np.array(all_labels, dtype=np.int64), np.array(all_preds, dtype=np.int64)


def log_confusion_matrix(cm: np.ndarray, class_names: Tuple[str, ...], prefix: str = "") -> None:
    """
    Keep your existing log format stable:
      "Confusion Matrix (rows=true, cols=pred):"
      "  Class <name>: [row]"
    """
    logger.info(prefix + "Confusion Matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        logger.info(prefix + f"  Class {class_names[i]:>15s}: {row}")


def train_one_run(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    run_name: str = "",
) -> float:
    """Train with early stopping on validation loss; return best validation loss."""
    best_val_loss = float("inf")
    epochs_wo_improve = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum, train_correct, n_train = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            n_train += bs
            train_loss_sum += float(loss.item()) * bs
            train_correct += int((logits.argmax(1) == y).sum().item())

        train_loss = train_loss_sum / max(1, n_train)
        train_acc = train_correct / max(1, n_train)

        model.eval()
        val_loss_sum, val_correct, n_val = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                n_val += bs
                val_loss_sum += float(loss.item()) * bs
                val_correct += int((logits.argmax(1) == y).sum().item())

        val_loss = val_loss_sum / max(1, n_val)
        val_acc = val_correct / max(1, n_val)

        if CFG.log_every_epoch:
            logger.info(
                f"{run_name} | Epoch {epoch}/{num_epochs} "
                f"| train_loss={train_loss:.4f}, train_acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_wo_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_wo_improve += 1
            if epochs_wo_improve >= patience:
                logger.info(f"{run_name} | Early stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return float(best_val_loss)


def evaluate_metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray, class_names: Tuple[str, ...]) -> Dict:
    """Compute multi-class metrics + confusion matrix + classification report."""
    num_classes = len(class_names)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
        "report": classification_report(y_true, y_pred, target_names=list(class_names), zero_division=0),
    }


# -------------------------
# Summary stats
# -------------------------

def t_critical_975(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
    }
    if df <= 0:
        return float("nan")
    return float(table.get(df, 1.96))


def mean_std_ci(values: List[float]) -> Tuple[float, float, Tuple[float, float]]:
    v = np.asarray(values, dtype=np.float64)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    n = len(v)
    if n > 1:
        half = t_critical_975(n - 1) * std / math.sqrt(n)
    else:
        half = 0.0
    return mean, std, (mean - half, mean + half)


# -------------------------
# Hyperopt space (your structured design)
# -------------------------

def build_search_space(cfg: Config) -> Dict:
    # Weighted bucket selection for weight decay
    wd_bucket = hp.pchoice("wd_bucket", [
        (cfg.wd_bucket_probs[0], "tiny"),
        (cfg.wd_bucket_probs[1], "small"),
        (cfg.wd_bucket_probs[2], "mid"),
        (cfg.wd_bucket_probs[3], "large"),
    ])

    space: Dict = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(3e-3)),
        "batch_size": hp.choice("batch_size", list(cfg.batch_size_choices)),
        "transformer_layers": hp.choice("transformer_layers", list(cfg.transformer_layers_choices)),
        "transformer_nhead": hp.choice("transformer_nhead", list(cfg.transformer_nhead_choices)),
        "num_conv_blocks": hp.choice("num_conv_blocks", list(cfg.num_conv_blocks_choices)),

        # weight decay bucket + per-bucket loguniform
        "wd_bucket": wd_bucket,
        "wd_tiny": hp.loguniform("wd_tiny", np.log(cfg.wd_tiny_range[0]), np.log(cfg.wd_tiny_range[1])),
        "wd_small": hp.loguniform("wd_small", np.log(cfg.wd_small_range[0]), np.log(cfg.wd_small_range[1])),
        "wd_mid": hp.loguniform("wd_mid", np.log(cfg.wd_mid_range[0]), np.log(cfg.wd_mid_range[1])),
        "wd_large": hp.loguniform("wd_large", np.log(cfg.wd_large_range[0]), np.log(cfg.wd_large_range[1])),

        # structured class-weight scaling
        "cw_global": hp.uniform("cw_global", cfg.cw_global_range[0], cfg.cw_global_range[1]),
    }

    # per-class deltas
    for i in range(len(cfg.artifact_labels)):
        space[f"cw_delta_{i}"] = hp.uniform(f"cw_delta_{i}", cfg.cw_delta_range[0], cfg.cw_delta_range[1])

    return space


def decode_best_params(best: Dict, cfg: Config) -> Dict:
    """
    Convert hyperopt 'best' into your stable output format:
      {'lr', 'batch_size', 'transformer_layers', 'transformer_nhead', 'num_conv_blocks', 'weight_decay', 'class_weight_scaling'}
    """
    # decode choices
    batch_size = list(cfg.batch_size_choices)[int(best["batch_size"])]
    transformer_layers = list(cfg.transformer_layers_choices)[int(best["transformer_layers"])]
    transformer_nhead = list(cfg.transformer_nhead_choices)[int(best["transformer_nhead"])]
    num_conv_blocks = list(cfg.num_conv_blocks_choices)[int(best["num_conv_blocks"])]

    # decode wd bucket
    wd_bucket = best["wd_bucket"]
    if wd_bucket == "tiny":
        weight_decay = float(best["wd_tiny"])
    elif wd_bucket == "small":
        weight_decay = float(best["wd_small"])
    elif wd_bucket == "mid":
        weight_decay = float(best["wd_mid"])
    else:
        weight_decay = float(best["wd_large"])

    # structured scaling
    cw_global = float(best["cw_global"])
    scaling = []
    for i in range(len(cfg.artifact_labels)):
        delta = float(best[f"cw_delta_{i}"])
        v = cw_global * (1.0 + delta)
        v = float(np.clip(v, cfg.cw_clip[0], cfg.cw_clip[1]))
        scaling.append(v)

    return {
        "lr": float(best["lr"]),
        "batch_size": int(batch_size),
        "transformer_layers": int(transformer_layers),
        "transformer_nhead": int(transformer_nhead),
        "num_conv_blocks": int(num_conv_blocks),
        "weight_decay": float(weight_decay),
        "class_weight_scaling": scaling,
    }


# -------------------------
# Main
# -------------------------

def main(cfg: Config) -> None:
    logger.info("########################")
    logger.info("model_2: Multi-class + Nested GroupKFold (subject-level) + verbose logs + confusion matrices")
    logger.info(f"DATA_DIR={cfg.data_dir}")
    logger.info(f"OUTER_K={cfg.outer_k}, INNER_K={cfg.inner_k}, MAX_EVALS={cfg.max_evals}")
    logger.info("########################")

    set_seed(cfg.seed)
    device = select_free_gpu()

    raw_samples = load_all_samples_with_subject(cfg.data_dir)
    logger.info(f"Total samples loaded (raw): {len(raw_samples)}")

    label_to_idx = {lbl: i for i, lbl in enumerate(cfg.artifact_labels)}
    dataset = EEGArtifactDatasetMulti(raw_samples, label_to_idx, eeg_scale=cfg.eeg_scale)

    groups = np.array([s["subject_id"] for s in dataset.samples])
    unique_subjects = sorted(set(groups.tolist()))
    logger.info(f"Unique subjects ({len(unique_subjects)}): {unique_subjects}")

    space = build_search_space(cfg)

    outer_cv = GroupKFold(n_splits=min(cfg.outer_k, len(unique_subjects)))
    outer_scores = defaultdict(list)

    for outer_fold, (trainval_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(dataset)), np.zeros(len(dataset)), groups=groups),
        start=1
    ):
        trainval_idx = np.array(trainval_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)

        trainval_groups = groups[trainval_idx]
        test_groups = groups[test_idx]

        logger.info("=" * 100)
        logger.info(f"[Outer Fold {outer_fold}/{outer_cv.n_splits}]")
        logger.info(f"TrainVal subjects ({len(set(trainval_groups))}): {sorted(set(trainval_groups.tolist()))}")
        logger.info(f"Test subjects     ({len(set(test_groups))}): {sorted(set(test_groups.tolist()))}")
        logger.info(f"TrainVal samples={len(trainval_idx)} | Test samples={len(test_idx)}")

        inner_n_splits = min(cfg.inner_k, len(set(trainval_groups.tolist())))
        if inner_n_splits < 2:
            raise ValueError("Not enough subjects in trainval split for inner GroupKFold. Reduce OUTER_K/INNER_K.")
        inner_cv = GroupKFold(n_splits=inner_n_splits)

        trial_counter = {"i": 0}
        best_seen = {"loss": float("inf")}

        def objective(params: Dict) -> Dict:
            trial_counter["i"] += 1
            t0 = time.time()

            lr = float(params["lr"])
            batch_size = int(params["batch_size"])
            transformer_layers = int(params["transformer_layers"])
            transformer_nhead = int(params["transformer_nhead"])
            num_conv_blocks = int(params["num_conv_blocks"])

            # select weight_decay based on bucket
            bucket = params["wd_bucket"]
            if bucket == "tiny":
                weight_decay = float(params["wd_tiny"])
            elif bucket == "small":
                weight_decay = float(params["wd_small"])
            elif bucket == "mid":
                weight_decay = float(params["wd_mid"])
            else:
                weight_decay = float(params["wd_large"])

            # class scaling: global*(1+delta), then clipped
            cw_global = float(params["cw_global"])
            class_scaling = []
            for i in range(NUM_CLASSES):
                delta = float(params[f"cw_delta_{i}"])
                v = cw_global * (1.0 + delta)
                v = float(np.clip(v, cfg.cw_clip[0], cfg.cw_clip[1]))
                class_scaling.append(v)

            base_filters = [8, 16, 32, 64, 128]
            cnn_filters = base_filters[:num_conv_blocks]

            # Transformer constraint
            if cnn_filters[-1] % transformer_nhead != 0:
                bad_loss = 1e9
                logger.info(
                    f"[Outer {outer_fold}] Trial {trial_counter['i']} INVALID (d_model%head!=0) => loss={bad_loss}"
                )
                return {"loss": bad_loss, "status": STATUS_OK}

            # Keep the original logging line (format stable)
            if cfg.log_trial_params:
                logger.info(f"[Outer {outer_fold}] Trial {trial_counter['i']}/{cfg.max_evals} params={params}")

            fold_losses = []
            agg_true, agg_pred = [], []

            for inner_fold, (inner_tr_rel, inner_va_rel) in enumerate(
                inner_cv.split(np.zeros(len(trainval_idx)), np.zeros(len(trainval_idx)), groups=trainval_groups),
                start=1
            ):
                inner_tr_idx = trainval_idx[inner_tr_rel]
                inner_va_idx = trainval_idx[inner_va_rel]

                base_w = compute_base_class_weights_maxnorm(dataset, inner_tr_idx, label_to_idx)
                adjusted_w = base_w * np.array(class_scaling, dtype=np.float32)
                class_w_tensor = torch.tensor(adjusted_w, dtype=torch.float32, device=device)

                tr_loader = DataLoader(
                    Subset(dataset, inner_tr_idx.tolist()),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                )
                va_loader = DataLoader(
                    Subset(dataset, inner_va_idx.tolist()),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                )

                model = CNNTransformerModel(
                    num_classes=NUM_CLASSES,
                    cnn_filters=cnn_filters,
                    transformer_layers=transformer_layers,
                    transformer_nhead=transformer_nhead,
                ).to(device)

                criterion = nn.CrossEntropyLoss(weight=class_w_tensor)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                run_name = f"[Outer {outer_fold}][Trial {trial_counter['i']}][Inner {inner_fold}]"
                val_loss = train_one_run(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=tr_loader,
                    val_loader=va_loader,
                    device=device,
                    num_epochs=cfg.inner_num_epochs,
                    patience=cfg.inner_patience,
                    run_name=run_name,
                )
                fold_losses.append(val_loss)

                y_true, y_pred = predict_on_loader(model, va_loader, device)
                agg_true.extend(y_true.tolist())
                agg_pred.extend(y_pred.tolist())

                if cfg.log_inner_fold_cm:
                    m = evaluate_metrics_multiclass(y_true, y_pred, cfg.artifact_labels)
                    logger.info(
                        f"{run_name} | val_loss={val_loss:.4f} | acc={m['acc']:.4f} | f1_macro={m['f1_macro']:.4f}"
                    )
                    log_confusion_matrix(m["cm"], cfg.artifact_labels, prefix=f"{run_name} | ")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            mean_loss = float(np.mean(fold_losses))
            dt = time.time() - t0
            best_seen["loss"] = min(best_seen["loss"], mean_loss)

            if cfg.log_trial_cm_agg and len(agg_true) > 0:
                agg_true_arr = np.array(agg_true, dtype=np.int64)
                agg_pred_arr = np.array(agg_pred, dtype=np.int64)
                m = evaluate_metrics_multiclass(agg_true_arr, agg_pred_arr, cfg.artifact_labels)

                logger.info(
                    f"[Outer {outer_fold}] Trial {trial_counter['i']} finished "
                    f"| inner_mean_loss={mean_loss:.6f} | agg_acc={m['acc']:.4f} | agg_f1_macro={m['f1_macro']:.4f} "
                    f"| best_loss_so_far={best_seen['loss']:.6f} | time={dt:.1f}s"
                )
                log_confusion_matrix(
                    m["cm"], cfg.artifact_labels, prefix=f"[Outer {outer_fold}] Trial {trial_counter['i']} AGG | "
                )
            else:
                logger.info(
                    f"[Outer {outer_fold}] Trial {trial_counter['i']} finished "
                    f"| inner_mean_loss={mean_loss:.6f} | best_loss_so_far={best_seen['loss']:.6f} | time={dt:.1f}s"
                )

            return {"loss": mean_loss, "status": STATUS_OK}

        logger.info(f"[Outer Fold {outer_fold}] Hyperopt tuning (inner CV mean val_loss) ...")
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=cfg.max_evals,
            trials=trials,
            rstate=np.random.default_rng(cfg.seed + outer_fold),
        )

        best_params = decode_best_params(best, cfg)
        logger.info(f"[Outer Fold {outer_fold}] Best hyperparams: {best_params}")

        # ----- Outer fold: train on full TrainVal, evaluate on Test -----
        base_filters = [8, 16, 32, 64, 128]
        cnn_filters = base_filters[:best_params["num_conv_blocks"]]

        base_w = compute_base_class_weights_maxnorm(dataset, trainval_idx, label_to_idx)
        adjusted_w = base_w * np.array(best_params["class_weight_scaling"], dtype=np.float32)
        class_w_tensor = torch.tensor(adjusted_w, dtype=torch.float32, device=device)

        train_loader = DataLoader(
            Subset(dataset, trainval_idx.tolist()),
            batch_size=best_params["batch_size"],
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx.tolist()),
            batch_size=best_params["batch_size"],
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        model = CNNTransformerModel(
            num_classes=NUM_CLASSES,
            cnn_filters=cnn_filters,
            transformer_layers=best_params["transformer_layers"],
            transformer_nhead=best_params["transformer_nhead"],
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_w_tensor)
        optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

        logger.info(f"[Outer Fold {outer_fold}] Final training on TrainVal, then evaluate on Test ...")
        _ = train_one_run(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=test_loader,  # preserve your original behavior
            device=device,
            num_epochs=cfg.final_num_epochs,
            patience=cfg.final_patience,
            run_name=f"[Outer {outer_fold}][FINAL]",
        )

        y_true, y_pred = predict_on_loader(model, test_loader, device)
        m = evaluate_metrics_multiclass(y_true, y_pred, cfg.artifact_labels)

        logger.info(f"[Outer Fold {outer_fold}] TEST metrics:")
        logger.info(f"  ACC      = {m['acc']:.4f}")
        logger.info(f"  BACC     = {m['bacc']:.4f}")
        logger.info(f"  F1_MACRO = {m['f1_macro']:.4f}")
        logger.info("  Classification Report:\n" + m["report"])

        if cfg.log_outer_test_cm:
            log_confusion_matrix(m["cm"], cfg.artifact_labels, prefix=f"[Outer Fold {outer_fold}] TEST | ")

        ckpt_path = f"model2_nested_groupkfold_fold{outer_fold}_model.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"[Outer Fold {outer_fold}] Saved: {ckpt_path}")

        outer_scores["acc"].append(float(m["acc"]))
        outer_scores["bacc"].append(float(m["bacc"]))
        outer_scores["f1_macro"].append(float(m["f1_macro"]))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Summary -----
    logger.info("=" * 100)
    logger.info("Nested GroupKFold Summary (Outer folds variability)")
    for key in ["acc", "bacc", "f1_macro"]:
        mean, std, ci = mean_std_ci(outer_scores[key])
        logger.info(f"{key.upper():>10s}: mean={mean:.4f}, std={std:.4f}, 95%CI=({ci[0]:.4f}, {ci[1]:.4f})")
    logger.info("-" * 100)
    for key in ["acc", "bacc", "f1_macro"]:
        logger.info(f"{key.upper():>10s} per-fold: {['{:.4f}'.format(x) for x in outer_scores[key]]}")

    logger.info("Done.")


if __name__ == "__main__":
    main(CFG)