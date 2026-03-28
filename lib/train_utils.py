import os
import numpy as np
from lib.metrics import evaluate_on_loader
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MetricHistory:
    """Store training metrics history"""
    loss: List[float]
    disc_f1: List[float]
    disc_p: List[float]
    disc_r: List[float]
    mae: List[float]
    mrr_err: List[float]
    prr50_err: List[float]
    sdrr_err: List[float]
    rmssd_err: List[float]

    def __init__(self):
        self.loss = []
        self.disc_f1 = []
        self.disc_p = []
        self.disc_r = []
        self.mae = []
        self.mrr_err = []
        self.prr50_err = []
        self.sdrr_err = []
        self.rmssd_err = []

    def append(self, loss: float, metrics: Dict):
        """Add metrics from one epoch"""
        self.loss.append(loss)
        self.disc_f1.append(metrics['disc_f1'])
        self.disc_p.append(metrics['disc_p'])
        self.disc_r.append(metrics['disc_r'])
        self.mae.append(metrics['mae'])
        self.mrr_err.append(metrics['mrr_err'])
        self.prr50_err.append(metrics['prr50_err'])
        self.sdrr_err.append(metrics['sdrr_err'])
        self.rmssd_err.append(metrics['rmssd_err'])


def compute_loss_on_loader(model, loader, criterion, device):
    """Compute average loss on a data loader"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for step, (x, y, ecg, subj_ids) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    return total_loss / (step + 1)


def evaluate_all_metrics(loader, model, device, eval_params):
    """Evaluate all metrics on a loader"""
    mae, _, mrr_err, prr50_err, sdrr_err, rmssd_err, disc_p, disc_r, disc_f1 = \
        evaluate_on_loader(
            loader, model, device,
            height_threshold=eval_params['height_threshold'],
            min_dist=eval_params['min_dist'],
            fs=eval_params['fs'],
            tol_ms=eval_params['tol_ms'],
            window_len=eval_params['window_len'],
            prominence=eval_params['prominence']
        )

    return {
        'mae': mae,
        'mrr_err': mrr_err,
        'prr50_err': prr50_err,
        'sdrr_err': sdrr_err,
        'rmssd_err': rmssd_err,
        'disc_p': disc_p,
        'disc_r': disc_r,
        'disc_f1': disc_f1
    }


def format_metric_line(split_name: str, metrics: Dict, compact: bool = False) -> str:
    """Format a single line of metrics for a data split"""
    if compact:
        return (
            f"{split_name:5s} | "
            f"P: {metrics['disc_p'] * 100:5.2f}%, "
            f"R: {metrics['disc_r'] * 100:5.2f}%, "
            f"F1: {metrics['disc_f1'] * 100:5.2f}%, "
            f"MAE: {metrics['mae']:.4f}"
        )
    else:
        return (
            f"{split_name:5s} | "
            f"P: {metrics['disc_p'] * 100:5.2f}%, "
            f"R: {metrics['disc_r'] * 100:5.2f}%, "
            f"F1: {metrics['disc_f1'] * 100:5.2f}%, "
            f"MAE: {metrics['mae']:.4f}, "
            f"mRR_err: {metrics['mrr_err']:.2f}%, "
        )


def train_model(model, trainloader, valloader, criterion, optimizer,
                scheduler, device, num_epochs, early_stop_patience, clip_norm,
                model_name, eval_params, save_dir, verbose: bool = True,
                min_epoch_improvement=3, log_fn=None):
    """Main training loop

    Args:
        model: PyTorch model to train
        trainloader: Training data loader
        valloader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Maximum number of epochs
        early_stop_patience: Number of epochs to wait for improvement
        clip_norm: Gradient clipping norm (0 to disable)
        model_name: Name for saving checkpoints
        eval_params: Dict with evaluation parameters
        save_dir: Directory to save the best model checkpoint
        verbose: If True, print all metrics; if False, compact output
        min_epoch_improvement: Minimum epoch before saving best model
        log_fn: Optional callable for streaming log lines to UI

    Returns:
        history: Dict with MetricHistory for train/val
        best_state: Dict with best model information
    """
    def _log(text):
        if log_fn is not None:
            log_fn(text)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best_model.pt")

    history = {
        'train': MetricHistory(),
        'val': MetricHistory(),
    }

    best_state = {
        'val_f1': 0,
        'epoch': 0,
        'patience_counter': 0
    }

    for epoch in range(num_epochs):
        # ============= Training Phase =============
        model.train()
        train_loss = 0.0
        for step, (x, y, ecg, subj_ids) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= (step + 1)

        # ============= Evaluation Phase =============
        val_loss = compute_loss_on_loader(model, valloader, criterion, device)

        train_metrics = evaluate_all_metrics(trainloader, model, device, eval_params)
        val_metrics = evaluate_all_metrics(valloader, model, device, eval_params)

        # ============= Update Histories =============
        history['train'].append(train_loss, train_metrics)
        history['val'].append(val_loss, val_metrics)

        # ============= Update Scheduler =============
        if not np.isnan(val_metrics['disc_f1']):
            scheduler.step(1 - val_metrics['disc_f1'])

        # ============= Log Metrics =============
        lr = optimizer.param_groups[0]['lr']
        header = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Losses  Tr: {train_loss:.4f}  V: {val_loss:.4f} | "
            f"LR: {lr:.2e}"
        )
        train_line = format_metric_line("Train", train_metrics, compact=not verbose)
        val_line = format_metric_line("Val", val_metrics, compact=not verbose)

        _log(header)
        _log(train_line)
        _log(val_line)

        # ============= Model Checkpointing =============
        is_improvement = (
            not np.isnan(val_metrics['disc_f1']) and
            val_metrics['disc_f1'] > best_state['val_f1']
        )

        if is_improvement and epoch >= min_epoch_improvement:
            best_state['val_f1'] = val_metrics['disc_f1']
            best_state['epoch'] = epoch
            best_state['patience_counter'] = 0
            torch.save(model.state_dict(), checkpoint_path)
            _log('+' * 50)
        else:
            best_state['patience_counter'] += 1
            is_improvement = False
            _log(f"{'-' * 40} {best_state['patience_counter']}/{early_stop_patience}")

        # ============= Early Stopping =============
        if best_state['patience_counter'] >= early_stop_patience:
            _log(f'Early stopping triggered at epoch {epoch+1}')
            break

    # ============= Load Best Model =============
    summary = (
        f"\nTraining completed. Best val F1: {best_state['val_f1']:.4f} "
        f"at epoch {best_state['epoch']+1}"
    )
    _log(summary)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    return history, best_state
