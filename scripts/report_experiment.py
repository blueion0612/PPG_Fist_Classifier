#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
final_report_experiment.py
--------------------------
Report experiment: produces the figures and result tables

Experiments:
    1. Effect of label and transition cleaning
    2. Calibration scenarios (zero-shot, few-shot, user-dependent)
    3. Baseline-subtracted features
    4. Final result table and figures
"""

import numpy as np
import pandas as pd
import glob
import json
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(REPO_ROOT, 'data', 'recordings')
FIGURES_DIR = os.path.join(REPO_ROOT, 'docs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Figures will be saved to: {FIGURES_DIR}")


# =============================================================================
# Data Loading
# =============================================================================

def load_ppg_data(csv_paths, key_channels=(1, 5, 7), window_sec=3.0, stride_sec=0.5,
                  fs=25.0, transition_guard_sec=1.0):
    """Load and window the recordings."""
    window_size = int(window_sec * fs)
    stride_size = int(stride_sec * fs)
    guard = int(transition_guard_sec * fs)

    nyq = 0.5 * fs
    b, a = butter(4, [0.5/nyq, 10.0/nyq], btype='band')

    all_windows = []
    all_labels = []
    all_sessions = []
    all_baselines = []
    all_indices = []  # position of each window in its recording

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        labels = df['label'].values
        channels = [f'ch{i:02d}' for i in range(16)]
        data = df[channels].values.astype(np.float32)

        valid_mask = np.isin(labels, [0, 1, 2, 3])
        labels = labels[valid_mask]
        data = data[valid_mask]
        binary_labels = (labels == 1).astype(int)

        # Transition guard
        transitions = np.where(np.diff(binary_labels) != 0)[0]
        trans_mask = np.ones(len(labels), dtype=bool)
        for t in transitions:
            trans_mask[max(0, t-guard):min(len(labels), t+guard+1)] = False

        data_key = data[:, list(key_channels)]

        for ch in range(data_key.shape[1]):
            data_key[:, ch] = filtfilt(b, a, data_key[:, ch])

        # Baseline from the first 10 seconds
        calib_end = min(int(10 * fs), len(data_key))
        baseline = data_key[:calib_end].mean(axis=0)
        baseline_std = data_key[:calib_end].std(axis=0)

        # Normalize
        for ch in range(data_key.shape[1]):
            mean_val = data_key[:calib_end, ch].mean()
            std_val = data_key[:calib_end, ch].std()
            if std_val < 1e-6:
                std_val = 1.0
            data_key[:, ch] = (data_key[:, ch] - mean_val) / std_val

        session_name = os.path.basename(csv_path).replace('.csv', '')

        for start in range(0, len(labels) - window_size + 1, stride_size):
            end = start + window_size
            if not np.all(trans_mask[start:end]):
                continue

            win_labels = binary_labels[start:end]
            label_ratio = win_labels.mean()

            if label_ratio >= 0.9:
                y = 1
            elif label_ratio <= 0.1:
                y = 0
            else:
                continue

            all_windows.append(data_key[start:end])
            all_labels.append(y)
            all_sessions.append(session_name)
            all_baselines.append(baseline)
            all_indices.append((csv_path, start, end))

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    sessions = np.array(all_sessions)
    baselines = np.array(all_baselines, dtype=np.float32)

    return X, y, sessions, baselines, all_indices


def apply_baseline_subtraction(X, baselines):
    """Subtract each recording own baseline from its windows."""
    X_new = X.copy()
    for i in range(len(X)):
        for ch in range(X.shape[2]):
            X_new[i, :, ch] = X[i, :, ch] - baselines[i, ch]
    return X_new


# =============================================================================
# Model
# =============================================================================

class MultiScaleCNN(nn.Module):
    """Multi-Scale CNN"""
    def __init__(self, n_channels=3, window_size=75):
        super().__init__()
        self.conv1_3 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
        self.conv1_7 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(96)

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.conv1_3(x)
        x2 = self.conv1_5(x)
        x3 = self.conv1_7(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = torch.relu(self.bn1(x))
        x = nn.MaxPool1d(2)(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).float()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                outputs = torch.sigmoid(model(X_batch))
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(y_batch.numpy())

        val_f1 = f1_score(val_labels, (np.array(val_preds) > 0.5).astype(int), zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def evaluate_model(model, test_loader):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = torch.sigmoid(model(X_batch))
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    }

    return metrics, all_probs, all_labels, all_preds


# =============================================================================
# Figure Generation
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Save a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Stable', 'Fist']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Cell annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_roc_curve(y_true, y_prob, title, filename):
    """Save an ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_precision_recall_curve(y_true, y_prob, title, filename):
    """Save a precision-recall curve figure."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_session_performance(session_metrics, title, filename):
    """Save the per-session F1 bar chart."""
    sessions = list(session_metrics.keys())
    f1_scores = [session_metrics[s]['f1'] for s in sessions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sessions, f1_scores, color='steelblue', edgecolor='black')

    # Mean line
    avg_f1 = np.mean(f1_scores)
    ax.axhline(y=avg_f1, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_f1:.3f}')

    ax.set_xlabel('Session')
    ax.set_ylabel('F1 Score')
    ax.set_title(title)
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bar labels
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{f1:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_calibration_curve(calib_results, title, filename):
    """Save the calibration-length sweep figure."""
    calib_lengths = list(calib_results.keys())
    f1_scores = [calib_results[c]['f1'] for c in calib_lengths]
    auc_scores = [calib_results[c]['auc'] for c in calib_lengths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(calib_lengths, f1_scores, 'bo-', linewidth=2, markersize=8, label='F1 Score')
    ax.plot(calib_lengths, auc_scores, 'rs-', linewidth=2, markersize=8, label='AUC')

    ax.set_xlabel('Calibration Length (seconds)')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_ylim([0.3, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_scenario_comparison(scenario_results, filename):
    """Save the scenario comparison figure."""
    scenarios = list(scenario_results.keys())
    f1_scores = [scenario_results[s]['f1'] for s in scenarios]
    auc_scores = [scenario_results[s]['auc'] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='steelblue')
    bars2 = ax.bar(x + width/2, auc_scores, width, label='AUC', color='coral')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bar labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def save_results_table(results_dict, filename):
    """Write the result table as CSV."""
    df = pd.DataFrame(results_dict).T
    df.to_csv(os.path.join(FIGURES_DIR, filename))
    print(f"  Saved: {filename}")
    return df


# =============================================================================
# Main Experiments
# =============================================================================

def main():
    print("=" * 80)
    print("FINAL REPORT EXPERIMENT")
    print("=" * 80)

    # Load data
    print("\n[1] Loading Data...")
    normal_files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, 'samples_*.csv')))
    tight_files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, 'tight', 'samples_*.csv')))
    all_files = normal_files + tight_files

    X, y, sessions, baselines, indices = load_ppg_data(all_files)
    print(f"  Data shape: {X.shape}")
    print(f"  Class balance: Stable={np.sum(y==0)}, Fist={np.sum(y==1)}")

    n_channels = X.shape[2]
    window_size = X.shape[1]
    unique_sessions = np.unique(sessions)

    all_results = {}

    # =========================================================================
    # Experiment 1: LOSO Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("[2] Experiment 1: LOSO Baseline (Zero-shot)")
    print("=" * 80)

    loso_metrics = {}
    all_probs_loso = []
    all_labels_loso = []

    for test_sess in unique_sessions:
        test_mask = sessions == test_sess
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=64, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=64
        )

        model = MultiScaleCNN(n_channels, window_size)
        model, _ = train_model(model, train_loader, test_loader, epochs=40, lr=1e-3)
        metrics, probs, labels, preds = evaluate_model(model, test_loader)

        loso_metrics[test_sess] = metrics
        all_probs_loso.extend(probs)
        all_labels_loso.extend(labels)

        print(f"    {test_sess}: F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

    avg_f1 = np.mean([m['f1'] for m in loso_metrics.values()])
    avg_auc = np.mean([m['auc'] for m in loso_metrics.values()])
    all_results['LOSO (Zero-shot)'] = {'f1': avg_f1, 'auc': avg_auc}
    print(f"\n  Average: F1={avg_f1:.3f}, AUC={avg_auc:.3f}")

    # Figures
    print("\n  Generating figures...")
    plot_session_performance(loso_metrics, 'LOSO Session Performance (Zero-shot)', 'fig1_loso_session_performance.png')
    plot_roc_curve(all_labels_loso, all_probs_loso, 'ROC Curve - LOSO (Zero-shot)', 'fig2_loso_roc_curve.png')
    plot_confusion_matrix(all_labels_loso, (np.array(all_probs_loso) > 0.5).astype(int),
                         'Confusion Matrix - LOSO (Zero-shot)', 'fig3_loso_confusion_matrix.png')

    # =========================================================================
    # Experiment 2: Within-Session (User-dependent)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3] Experiment 2: Within-Session (User-dependent)")
    print("=" * 80)

    within_metrics = {}

    for sess in unique_sessions:
        mask = sessions == sess
        X_sess, y_sess = X[mask], y[mask]

        if len(np.unique(y_sess)) < 2 or len(y_sess) < 50:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_sess, y_sess, test_size=0.2, random_state=42, stratify=y_sess
        )

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=64, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=64
        )

        model = MultiScaleCNN(n_channels, window_size)
        model, _ = train_model(model, train_loader, test_loader, epochs=30, lr=1e-3)
        metrics, _, _, _ = evaluate_model(model, test_loader)

        within_metrics[sess] = metrics
        print(f"    {sess}: F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

    avg_f1_within = np.mean([m['f1'] for m in within_metrics.values()])
    avg_auc_within = np.mean([m['auc'] for m in within_metrics.values()])
    all_results['Within-Session (User-dep)'] = {'f1': avg_f1_within, 'auc': avg_auc_within}
    print(f"\n  Average: F1={avg_f1_within:.3f}, AUC={avg_auc_within:.3f}")

    plot_session_performance(within_metrics, 'Within-Session Performance (User-dependent)',
                            'fig4_within_session_performance.png')

    # =========================================================================
    # Experiment 3: Few-shot Calibration
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4] Experiment 3: Few-shot Calibration Scenarios")
    print("=" * 80)

    calib_lengths = [10, 20, 30, 40, 60]  # seconds
    calib_results = {}
    fs = 25.0

    for calib_sec in calib_lengths:
        calib_samples = int(calib_sec * fs / 0.5)  # windows (stride 0.5s)
        fewshot_f1s = []
        fewshot_aucs = []

        for test_sess in unique_sessions:
            test_mask = sessions == test_sess
            train_mask = ~test_mask

            X_train_global = X[train_mask]
            y_train_global = y[train_mask]
            X_test_sess = X[test_mask]
            y_test_sess = y[test_mask]

            if len(X_test_sess) < calib_samples + 20:
                continue
            if len(np.unique(y_train_global)) < 2 or len(np.unique(y_test_sess)) < 2:
                continue

            # Split test session: calibration + actual test
            X_calib = X_test_sess[:calib_samples]
            y_calib = y_test_sess[:calib_samples]
            X_test = X_test_sess[calib_samples:]
            y_test = y_test_sess[calib_samples:]

            if len(np.unique(y_calib)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # Combine global + calibration
            X_train_combined = np.concatenate([X_train_global, X_calib], axis=0)
            y_train_combined = np.concatenate([y_train_global, y_calib], axis=0)

            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train_combined), torch.LongTensor(y_train_combined)),
                batch_size=64, shuffle=True
            )
            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
                batch_size=64
            )

            model = MultiScaleCNN(n_channels, window_size)
            model, _ = train_model(model, train_loader, test_loader, epochs=30, lr=1e-3)
            metrics, _, _, _ = evaluate_model(model, test_loader)

            fewshot_f1s.append(metrics['f1'])
            fewshot_aucs.append(metrics['auc'])

        if fewshot_f1s:
            avg_f1 = np.mean(fewshot_f1s)
            avg_auc = np.mean(fewshot_aucs)
            calib_results[calib_sec] = {'f1': avg_f1, 'auc': avg_auc}
            all_results[f'Few-shot ({calib_sec}s)'] = {'f1': avg_f1, 'auc': avg_auc}
            print(f"    Calibration {calib_sec}s: F1={avg_f1:.3f}, AUC={avg_auc:.3f}")

    if calib_results:
        plot_calibration_curve(calib_results, 'Calibration Length vs Performance',
                              'fig5_calibration_curve.png')

    # =========================================================================
    # Experiment 4: Baseline Subtraction
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5] Experiment 4: Baseline Subtraction Feature")
    print("=" * 80)

    X_baseline = apply_baseline_subtraction(X, baselines)
    baseline_f1s = []

    for test_sess in unique_sessions:
        test_mask = sessions == test_sess
        train_mask = ~test_mask

        X_train, X_test = X_baseline[train_mask], X_baseline[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=64, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=64
        )

        model = MultiScaleCNN(n_channels, window_size)
        model, _ = train_model(model, train_loader, test_loader, epochs=40, lr=1e-3)
        metrics, _, _, _ = evaluate_model(model, test_loader)

        baseline_f1s.append(metrics['f1'])
        print(f"    {test_sess}: F1={metrics['f1']:.3f}")

    avg_f1_baseline = np.mean(baseline_f1s)
    all_results['LOSO + Baseline Sub'] = {'f1': avg_f1_baseline, 'auc': 0.0}
    print(f"\n  Average with Baseline Subtraction: F1={avg_f1_baseline:.3f}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Save results table
    df = save_results_table(all_results, 'results_table.csv')
    print("\n" + df.to_string())

    # Scenario comparison figure
    plot_scenario_comparison(all_results, 'fig6_scenario_comparison.png')

    # Machine-readable copy of every number the README states. docs/figures/make_hero.py
    # draws from it and tests/test_readme_numbers.py checks the README against it.
    with open(os.path.join(FIGURES_DIR, 'results.json'), 'w', encoding='utf-8') as fh:
        json.dump({
            '_source': 'scripts/report_experiment.py',
            'scenarios': {k: {'f1': round(float(v['f1']), 3), 'auc': round(float(v['auc']), 3)}
                          for k, v in all_results.items()},
            'loso_session_f1': {str(s): round(float(m['f1']), 3) for s, m in loso_metrics.items()},
            'loso_mean_f1': round(float(all_results['LOSO (Zero-shot)']['f1']), 3),
        }, fh, indent=2)

    # Summary statistics
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"""
    1. Zero-shot (LOSO): F1 = {all_results.get('LOSO (Zero-shot)', {}).get('f1', 0):.3f}
       - generic model applied to an unseen session

    2. User-dependent: F1 = {all_results.get('Within-Session (User-dep)', {}).get('f1', 0):.3f}
       - upper bound for a per-session model (gap: {all_results.get('Within-Session (User-dep)', {}).get('f1', 0) - all_results.get('LOSO (Zero-shot)', {}).get('f1', 0):.3f})

    3. Few-shot Calibration:
       - performance against calibration length
       - use this to pick a calibration duration

    4. Baseline Subtraction: F1 = {avg_f1_baseline:.3f}
       - effect of correcting the per-session baseline
    """)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
