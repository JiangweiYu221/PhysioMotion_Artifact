#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nested subject-independent GroupKFold training with Hyperopt for EEG artifact binary classification.

Key features
- Subject-level nested CV (outer = evaluation, inner = hyperparameter tuning)
- Hyperopt (TPE) tuning on inner-fold mean validation loss
- Robust CUDA OOM handling (a trial that OOMs is assigned a very bad loss and skipped)
- Detailed logging to both console and a log file

Notes
- This script assumes each .pkl file contains a list of dicts, each dict includes:
  - "eeg_data": array-like shaped (C, T) (e.g., 34 x 375)
  - "artifact_types": list[str], e.g. ["close_base"] or ["blink", ...]
- Labels are mapped to a binary task:
  - ["close_base"] -> "close_base"
  - otherwise -> "artifact"
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
    precision_score,
    recall_score,
)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

try:
    import GPUtil
except Exception:
    GPUtil = None

# -------------------------
# Configuration
# -------------------------

@dataclass
class Config:
    # Data
    data_dir: str = "/home/mnt_disk1/Motion_Artifact_han1/Slidingwindow"
    top2_labels: Tuple[str, str] = ("close_base", "artifact")

    # Nested CV
    outer_k: int = 5
    inner_k: int = 3
    max_evals: int = 30
    seed: int = 42

    # Training
    final_num_epochs: int = 15
    final_patience: int = 3
    inner_num_epochs: int = 10
    inner_patience: int = 2

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True

    # Logging
    log_filename: str = "nested_groupkfold_verbose_log.txt"
    log_every_epoch: bool = True
    log_inner_fold_detail: bool = True
    log_trial_params: bool = True
    show_tqdm: bool = False  # keep False for cleaner logs

    # GPU selection / OOM safety
    min_free_mb: int = 6000
    oom_bad_loss: float = 1e9

    # Hyperopt search space (OOM-safer default)
    # (You can widen it, but OOM risk increases.)
    batch_size_choices: Tuple[int, ...] = (64, 128)
    transformer_layers_choices: Tuple[int, ...] = (1, 2)
    transformer_nhead_choices: Tuple[int, ...] = (2, 4, 8)
    num_conv_blocks_choices: Tuple[int, ...] = (3, 4, 5)

    # Class-weight scaling search ranges
    cw0_low: float = 0.5
    cw0_high: float = 0.7
    cw1_low: float = 1.1
    cw1_high: float = 1.4


CFG = Config()

warnings.filterwarnings("ignore")


# -------------------------
# Logging utilities
# -------------------------

def setup_logger(log_filename: str) -> logging.Logger:
    logger = logging.getLogger("nested_cv")
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
    # Determinism (may reduce throughput)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# GPU selection & cleanup
# -------------------------

def select_free_gpu(min_free_mb: int = 6000) -> torch.device:
    """
    Select the GPU with the largest free memory.
    If free memory is below threshold, fall back to CPU (safer for long tuning runs).
    """
    if (GPUtil is None) or (not torch.cuda.is_available()):
        logger.info("GPUtil not available or CUDA not available; using CPU.")
        return torch.device("cpu")

    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        logger.info("No GPUs detected by GPUtil; using CPU.")
        return torch.device("cpu")

    best = max(gpus, key=lambda g: g.memoryFree)
    logger.info("GPU candidates: " + " | ".join([f"id={g.id}, free={g.memoryFree:.0f}MB" for g in gpus]))
    logger.info(f"Selected GPU: id={best.id} name={best.name}, free={best.memoryFree:.0f}MB")

    if best.memoryFree < min_free_mb:
        logger.warning(
            f"Best GPU free memory {best.memoryFree:.0f}MB < {min_free_mb}MB. "
            f"Falling back to CPU to reduce OOM risk. (You may lower min_free_mb to force GPU.)"
        )
        return torch.device("cpu")

    return torch.device(f"cuda:{best.id}")


def cleanup_cuda() -> None:
    """Best-effort CUDA cache cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# -------------------------
# Data loading
# -------------------------

SUBJECT_RE = re.compile(r"(sub\d+)_run\d+\.pkl$", re.IGNORECASE)


def parse_subject_id(filename: str) -> str:
    """
    Parse subject id from filename like:
      sub1_run01.pkl -> "sub1"
    """
    m = SUBJECT_RE.search(filename)
    if not m:
        raise ValueError(
            f"Cannot parse subject_id from filename: {filename}. Expected pattern like sub1_run01.pkl"
        )
    return m.group(1).lower()


def load_all_samples_with_subject(data_dir: str) -> List[dict]:
    """
    Load all samples from all .pkl files under data_dir.
    Adds 'subject_id' into each sample dict.
    """
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

        # arr is expected to be list[dict]
        for s in arr:
            if isinstance(s, dict) and ("eeg_data" in s) and ("artifact_types" in s):
                d = dict(s)
                d["subject_id"] = subject_id
                samples.append(d)

    return samples


# -------------------------
# Dataset
# -------------------------

class EEGArtifactDataset(Dataset):
    """
    Binary dataset:
      label = "close_base" if artifact_types == ["close_base"] else "artifact"
    """

    def __init__(self, samples: List[dict], label_to_idx: Dict[str, int], scale: float = 1e6):
        super().__init__()
        self.label_to_idx = label_to_idx
        self.scale = float(scale)

        filtered = []
        for s in samples:
            if ("eeg_data" not in s) or ("artifact_types" not in s) or ("subject_id" not in s):
                continue

            label = "close_base" if s["artifact_types"] == ["close_base"] else "artifact"
            if label not in label_to_idx:
                continue

            d = dict(s)
            d["label"] = label
            filtered.append(d)

        self.samples = filtered

        logger.info(f"Filtered dataset size: {len(self.samples)}")
        label_counter = Counter([s["label"] for s in self.samples])
        subj_counter = Counter([s["subject_id"] for s in self.samples])
        logger.info(f"Label distribution: {label_counter}")
        logger.info(
            f"Num subjects: {len(subj_counter)} | "
            f"subject sample range: min={min(subj_counter.values())}, max={max(subj_counter.values())}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = np.asarray(s["eeg_data"], dtype=np.float32) * self.scale  # scale to microvolts
        y = self.label_to_idx[s["label"]]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# -------------------------
# Model
# -------------------------

class CNNTransformerModel(nn.Module):
    """
    CNN + Transformer encoder baseline.
    Input: (B, C, T) -> treated as (B, 1, C, T) to use 2D conv.
    """

    def __init__(
        self,
        num_classes: int = 2,
        cnn_filters: List[int] = None,
        transformer_nhead: int = 8,
        transformer_ff: int = 1024,
        transformer_layers: int = 1,
        dropout_p: float = 0.5,
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
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*convs)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cnn_filters[-1],
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(cnn_filters[-1], 100)
        self.fc2 = nn.Linear(100, 100)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)            # (B,1,C,T)
        x = self.cnn(x)               # (B,Ch,H,W)
        x = x.flatten(2).transpose(1, 2)  # (B,seq,Ch)
        x = self.transformer(x)       # (B,seq,Ch)
        x = x.transpose(1, 2)         # (B,Ch,seq)
        x = self.global_pool(x).squeeze(-1)  # (B,Ch)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


# -------------------------
# Training / evaluation
# -------------------------

def compute_class_weights_from_indices(
    dataset: EEGArtifactDataset,
    indices: np.ndarray,
    label_to_idx: Dict[str, int]
) -> np.ndarray:
    """Compute inverse-frequency weights from a subset of indices."""
    counts = np.zeros(len(label_to_idx), dtype=np.float64)
    for i in indices:
        lbl = dataset.samples[int(i)]["label"]
        counts[label_to_idx[lbl]] += 1.0
    counts = np.maximum(counts, 1.0)
    total = float(counts.sum())
    weights = total / counts
    return weights.astype(np.float32)


def train_one_run(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    run_name: str,
    log_every_epoch: bool,
) -> Tuple[float, float]:
    """
    Train with early stopping on validation loss.
    Returns: (best_val_loss, best_val_acc_at_end_epoch)
    """
    best_val_loss = float("inf")
    epochs_wo_improve = 0
    best_state = None
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_correct += int((logits.argmax(1) == y).sum().item())
            train_n += bs

        avg_train_loss = train_loss_sum / max(1, train_n)
        train_acc = train_correct / max(1, train_n)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_correct += int((logits.argmax(1) == y).sum().item())
                val_n += bs

        avg_val_loss = val_loss_sum / max(1, val_n)
        val_acc = val_correct / max(1, val_n)

        if log_every_epoch:
            logger.info(
                f"{run_name} | Epoch {epoch}/{num_epochs} "
                f"| train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f} "
                f"| val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            epochs_wo_improve = 0
            # Save CPU copy for safety
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_wo_improve += 1
            if epochs_wo_improve >= patience:
                if log_every_epoch:
                    logger.info(f"{run_name} | Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return float(best_val_loss), float(best_val_acc)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, labels: List[str]) -> Dict:
    """Evaluate on loader and return standard classification metrics + confusion matrix."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            trues.extend(y.numpy().tolist())

    y_true = np.array(trues, dtype=np.int64)
    y_pred = np.array(preds, dtype=np.int64)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=labels, zero_division=0),
    }


# -------------------------
# Summary stats (mean/std/95% CI)
# -------------------------

def t_critical_975(df: int) -> float:
    """
    Two-sided 95% CI critical t value (0.975 quantile).
    For df > 30, use normal approximation 1.96.
    """
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
        tcrit = t_critical_975(n - 1)
        half = tcrit * std / math.sqrt(n)
    else:
        half = 0.0
    return mean, std, (mean - half, mean + half)


# -------------------------
# Hyperopt search space
# -------------------------

def build_search_space(cfg: Config) -> Dict:
    """
    Hyperopt search space. Keep it relatively safe to avoid OOM.
    """
    return {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "batch_size": hp.choice("batch_size", list(cfg.batch_size_choices)),
        "transformer_layers": hp.choice("transformer_layers", list(cfg.transformer_layers_choices)),
        "transformer_nhead": hp.choice("transformer_nhead", list(cfg.transformer_nhead_choices)),
        "num_conv_blocks": hp.choice("num_conv_blocks", list(cfg.num_conv_blocks_choices)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-3)),
        "class_weight_scaling_0": hp.uniform("class_weight_scaling_0", cfg.cw0_low, cfg.cw0_high),
        "class_weight_scaling_1": hp.uniform("class_weight_scaling_1", cfg.cw1_low, cfg.cw1_high),
    }


def decode_best_params(best: Dict, cfg: Config) -> Dict:
    """Decode hp.choice indices into actual values."""
    bs_choices = list(cfg.batch_size_choices)
    lyr_choices = list(cfg.transformer_layers_choices)
    head_choices = list(cfg.transformer_nhead_choices)
    conv_choices = list(cfg.num_conv_blocks_choices)

    return {
        "lr": float(best["lr"]),
        "batch_size": int(bs_choices[int(best["batch_size"])]),
        "transformer_layers": int(lyr_choices[int(best["transformer_layers"])]),
        "transformer_nhead": int(head_choices[int(best["transformer_nhead"])]),
        "num_conv_blocks": int(conv_choices[int(best["num_conv_blocks"])]),
        "weight_decay": float(best["weight_decay"]),
        "class_weight_scaling": [float(best["class_weight_scaling_0"]), float(best["class_weight_scaling_1"])],
    }


# -------------------------
# Main: nested CV
# -------------------------

def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = select_free_gpu(min_free_mb=cfg.min_free_mb)

    logger.info("########################")
    logger.info("Nested GroupKFold subject-level CV (OOM-safe)")
    logger.info(f"DATA_DIR={cfg.data_dir}")
    logger.info(f"OUTER_K={cfg.outer_k}, INNER_K={cfg.inner_k}, MAX_EVALS={cfg.max_evals}")
    logger.info("Tip: to reduce CUDA fragmentation, you may set:")
    logger.info("     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    logger.info("########################")

    raw_samples = load_all_samples_with_subject(cfg.data_dir)
    logger.info(f"Total samples loaded (raw): {len(raw_samples)}")

    label_to_idx = {lbl: i for i, lbl in enumerate(cfg.top2_labels)}
    dataset = EEGArtifactDataset(raw_samples, label_to_idx)

    groups = np.array([s["subject_id"] for s in dataset.samples])
    unique_subjects = sorted(set(groups.tolist()))
    logger.info(f"Unique subjects ({len(unique_subjects)}): {unique_subjects}")

    space = build_search_space(cfg)

    outer_cv = GroupKFold(n_splits=min(cfg.outer_k, len(unique_subjects)))
    outer_metrics = defaultdict(list)

    for outer_fold, (trainval_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(dataset)), np.zeros(len(dataset)), groups=groups),
        start=1
    ):
        trainval_idx = np.array(trainval_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)

        trainval_groups = groups[trainval_idx]
        test_groups = groups[test_idx]

        logger.info("=" * 90)
        logger.info(f"[Outer Fold {outer_fold}/{outer_cv.n_splits}]")
        logger.info(f"TrainVal subjects ({len(set(trainval_groups))}): {sorted(set(trainval_groups.tolist()))}")
        logger.info(f"Test subjects     ({len(set(test_groups))}): {sorted(set(test_groups.tolist()))}")
        logger.info(f"TrainVal samples={len(trainval_idx)} | Test samples={len(test_idx)}")

        inner_n_splits = min(cfg.inner_k, len(set(trainval_groups.tolist())))
        if inner_n_splits < 2:
            raise ValueError("Not enough subjects in TrainVal split for inner GroupKFold. Reduce OUTER_K/INNER_K.")

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
            weight_decay = float(params["weight_decay"])
            class_weight_scaling = [float(params["class_weight_scaling_0"]), float(params["class_weight_scaling_1"])]

            base_filters = [8, 16, 32, 64, 128]
            cnn_filters = base_filters[:num_conv_blocks]

            # Constraint: d_model must be divisible by nhead
            if cnn_filters[-1] % transformer_nhead != 0:
                logger.info(
                    f"[Outer {outer_fold}] Trial {trial_counter['i']} INVALID (d_model%head!=0) => loss={cfg.oom_bad_loss}"
                )
                return {"loss": cfg.oom_bad_loss, "status": STATUS_OK}

            if cfg.log_trial_params:
                logger.info(f"[Outer {outer_fold}] Trial {trial_counter['i']}/{cfg.max_evals} params={params}")

            fold_losses = []
            fold_accs = []

            try:
                for inner_fold, (inner_tr_rel, inner_va_rel) in enumerate(
                    inner_cv.split(np.zeros(len(trainval_idx)), np.zeros(len(trainval_idx)), groups=trainval_groups),
                    start=1
                ):
                    inner_tr_idx = trainval_idx[inner_tr_rel]
                    inner_va_idx = trainval_idx[inner_va_rel]

                    # class weights computed from inner-train only
                    base_w = compute_class_weights_from_indices(dataset, inner_tr_idx, label_to_idx)
                    adjusted = base_w * np.array(class_weight_scaling, dtype=np.float32)
                    class_w_tensor = torch.tensor(adjusted, dtype=torch.float32, device=device)

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
                        num_classes=len(label_to_idx),
                        cnn_filters=cnn_filters,
                        transformer_layers=transformer_layers,
                        transformer_nhead=transformer_nhead,
                    ).to(device)

                    criterion = nn.CrossEntropyLoss(weight=class_w_tensor)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    run_name = f"[Outer {outer_fold}][Trial {trial_counter['i']}][Inner {inner_fold}]"
                    val_loss, val_acc = train_one_run(
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        train_loader=tr_loader,
                        val_loader=va_loader,
                        device=device,
                        num_epochs=cfg.inner_num_epochs,
                        patience=cfg.inner_patience,
                        run_name=run_name,
                        log_every_epoch=cfg.log_every_epoch,
                    )

                    fold_losses.append(val_loss)
                    fold_accs.append(val_acc)

                    if cfg.log_inner_fold_detail:
                        subs_tr = sorted(set(trainval_groups[inner_tr_rel].tolist()))
                        subs_va = sorted(set(trainval_groups[inner_va_rel].tolist()))
                        logger.info(
                            f"{run_name} | done | val_loss={val_loss:.4f}, val_acc={val_acc:.4f} "
                            f"| tr_subs={subs_tr} | va_subs={subs_va}"
                        )

                    # release memory
                    del model, optimizer, criterion, tr_loader, va_loader
                    cleanup_cuda()

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(
                    f"[Outer {outer_fold}] Trial {trial_counter['i']} OOM => mark as bad loss. "
                    f"err={str(e)[:160]}"
                )
                cleanup_cuda()
                dt = time.time() - t0
                logger.info(f"[Outer {outer_fold}] Trial {trial_counter['i']} finished (OOM) | time={dt:.1f}s")
                return {"loss": cfg.oom_bad_loss, "status": STATUS_OK}

            mean_loss = float(np.mean(fold_losses)) if len(fold_losses) > 0 else cfg.oom_bad_loss
            mean_acc = float(np.mean(fold_accs)) if len(fold_accs) > 0 else 0.0
            dt = time.time() - t0

            best_seen["loss"] = min(best_seen["loss"], mean_loss)

            logger.info(
                f"[Outer {outer_fold}] Trial {trial_counter['i']} finished "
                f"| inner_mean_loss={mean_loss:.6f}, inner_mean_acc={mean_acc:.4f} "
                f"| best_loss_so_far={best_seen['loss']:.6f} | time={dt:.1f}s"
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

        # ----- Final training: trainval -> test evaluation -----
        base_filters = [8, 16, 32, 64, 128]
        cnn_filters = base_filters[:best_params["num_conv_blocks"]]

        base_w = compute_class_weights_from_indices(dataset, trainval_idx, label_to_idx)
        adjusted = base_w * np.array(best_params["class_weight_scaling"], dtype=np.float32)
        class_w_tensor = torch.tensor(adjusted, dtype=torch.float32, device=device)

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
            num_classes=len(label_to_idx),
            cnn_filters=cnn_filters,
            transformer_layers=best_params["transformer_layers"],
            transformer_nhead=best_params["transformer_nhead"],
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_w_tensor)
        optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

        logger.info(f"[Outer Fold {outer_fold}] Final training on TrainVal, then evaluate on Test ...")

        try:
            _ = train_one_run(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=test_loader,  # keep your original behavior
                device=device,
                num_epochs=cfg.final_num_epochs,
                patience=cfg.final_patience,
                run_name=f"[Outer {outer_fold}][FINAL]",
                log_every_epoch=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[Outer Fold {outer_fold}] FINAL training OOM. err={str(e)[:200]}")
            cleanup_cuda()
            # minimal retry strategy: smaller batch size
            retry_bs = 32
            logger.warning(f"[Outer Fold {outer_fold}] Retry FINAL with batch_size={retry_bs} ...")
            train_loader = DataLoader(
                Subset(dataset, trainval_idx.tolist()),
                batch_size=retry_bs,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )
            test_loader = DataLoader(
                Subset(dataset, test_idx.tolist()),
                batch_size=retry_bs,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )
            optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
            _ = train_one_run(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=test_loader,
                device=device,
                num_epochs=cfg.final_num_epochs,
                patience=cfg.final_patience,
                run_name=f"[Outer {outer_fold}][FINAL][RETRY_BS32]",
                log_every_epoch=True,
            )

        metrics = evaluate_model(model, test_loader, device, labels=list(cfg.top2_labels))
        logger.info(f"[Outer Fold {outer_fold}] TEST metrics:")
        logger.info(f"  ACC  = {metrics['acc']:.4f}")
        logger.info(f"  BACC = {metrics['bacc']:.4f}")
        logger.info(f"  F1   = {metrics['f1']:.4f}")
        logger.info(f"  Prec = {metrics['precision']:.4f}")
        logger.info(f"  Rec  = {metrics['recall']:.4f}")
        logger.info("  Classification Report:\n" + metrics["report"])
        logger.info("  Confusion Matrix:\n" + str(metrics["cm"]))

        ckpt_path = f"nested_groupkfold_verbose_fold{outer_fold}_model.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"[Outer Fold {outer_fold}] Saved model checkpoint: {ckpt_path}")

        outer_metrics["acc"].append(float(metrics["acc"]))
        outer_metrics["bacc"].append(float(metrics["bacc"]))
        outer_metrics["f1"].append(float(metrics["f1"]))
        outer_metrics["precision"].append(float(metrics["precision"]))
        outer_metrics["recall"].append(float(metrics["recall"]))

        del model, optimizer, criterion, train_loader, test_loader
        cleanup_cuda()

    # ----- Summary -----
    logger.info("=" * 90)
    logger.info("Nested GroupKFold Summary (Outer-fold variability)")
    for k in ["acc", "bacc", "f1", "precision", "recall"]:
        mean, std, ci = mean_std_ci(outer_metrics[k])
        logger.info(f"{k.upper():>9s}: mean={mean:.4f}, std={std:.4f}, 95%CI=({ci[0]:.4f}, {ci[1]:.4f})")
    logger.info("-" * 90)
    for k in ["acc", "bacc", "f1", "precision", "recall"]:
        logger.info(f"{k.upper():>9s} per-fold: {['{:.4f}'.format(x) for x in outer_metrics[k]]}")

    logger.info("Done.")


if __name__ == "__main__":
    main(CFG)