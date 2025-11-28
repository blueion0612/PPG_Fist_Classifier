#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
PPG Fist Classifier 성능 평가

기능:
    1. 단순 Train/Test 분할 평가
    2. Leave-One-Session-Out (LOSO) 교차검증
    3. 상세 성능 리포트 (Accuracy, F1, Precision, Recall, AUC)
    4. Confusion Matrix 시각화

사용법:
    # 단순 평가
    python evaluate.py --data ./data/baseline_all.npz --model ./production_models/final_model_gb.pkl

    # LOSO 교차검증
    python evaluate.py --data ./data/baseline_all.npz --loso --model-type gb
"""

import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

from model import ModelConfig, ModelFactory, ModelPackage


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """NPZ 데이터 로드"""
    data = np.load(data_path, allow_pickle=True)

    X = data['X']
    y = data['y']
    sessions = data['session'] if 'session' in data else None

    return X, y, sessions


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """성능 지표 계산"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Performance"):
    """성능 지표 출력"""
    print(f"\n{title}")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Confusion Matrix 출력"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"              Predicted")
    print(f"              0 (Stable)  1 (Fist)")
    print(f"  Actual 0    {cm[0, 0]:^10}  {cm[0, 1]:^10}")
    print(f"  Actual 1    {cm[1, 0]:^10}  {cm[1, 1]:^10}")

    # Additional stats
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")


# =============================================================================
# Evaluation Methods
# =============================================================================

def evaluate_model(model_path: str, data_path: str, test_size: float = 0.2):
    """저장된 모델 평가 (Train/Test 분할)"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (Train/Test Split)")
    print("=" * 70)

    # Load data
    X, y, sessions = load_data(data_path)
    print(f"\n[Data]")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {dict(Counter(y))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\n[Split]")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Test:  {len(y_test)} samples")

    # Load model
    package = ModelPackage.load(model_path)
    print(f"\n[Model]")
    print(f"  Type: {package.config.model_type}")

    # Predict
    y_pred = package.predict(X_test)
    y_prob = package.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, "Test Performance")
    print_confusion_matrix(y_test, y_pred)

    return metrics


def evaluate_loso(data_path: str, model_type: str = 'gb',
                  learning_rate: float = 0.1, max_depth: int = 5):
    """Leave-One-Session-Out 교차검증"""
    print("\n" + "=" * 70)
    print("LEAVE-ONE-SESSION-OUT CROSS-VALIDATION")
    print("=" * 70)

    # Load data
    X, y, sessions = load_data(data_path)

    if sessions is None:
        print("Error: No session information in data")
        return None

    unique_sessions = np.unique(sessions)
    print(f"\n[Data]")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Sessions: {list(unique_sessions)}")

    # Config
    config = ModelConfig(
        model_type=model_type,
        learning_rate=learning_rate,
        max_depth=max_depth
    )

    print(f"\n[Model Configuration]")
    print(f"  Type: {model_type}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Depth: {max_depth}")

    # LOSO CV
    all_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    print(f"\n[Cross-Validation]")
    print("-" * 70)

    for test_session in unique_sessions:
        # Split
        test_mask = sessions == test_session
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  {test_session}: Skipped (insufficient classes)")
            continue

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = ModelFactory.create(config)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        all_metrics.append(metrics)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        print(f"  {test_session}: Acc={metrics['accuracy']:.3f}, "
              f"F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f} "
              f"(n={len(y_test)})")

    print("-" * 70)

    # Overall metrics
    if len(all_metrics) > 0:
        # Average across folds
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        std_metrics = {
            key: np.std([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }

        print(f"\n[Average Across Folds]")
        print("-" * 40)
        print(f"  Accuracy:  {avg_metrics['accuracy']:.4f} (+/- {std_metrics['accuracy']:.4f})")
        print(f"  F1 Score:  {avg_metrics['f1']:.4f} (+/- {std_metrics['f1']:.4f})")
        print(f"  Precision: {avg_metrics['precision']:.4f} (+/- {std_metrics['precision']:.4f})")
        print(f"  Recall:    {avg_metrics['recall']:.4f} (+/- {std_metrics['recall']:.4f})")
        print(f"  AUC:       {avg_metrics['auc']:.4f} (+/- {std_metrics['auc']:.4f})")

        # Overall confusion matrix
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        print_confusion_matrix(all_y_true, all_y_pred)

        return avg_metrics

    return None


def quick_benchmark(data_path: str):
    """여러 모델 빠른 벤치마크"""
    print("\n" + "=" * 70)
    print("QUICK MODEL BENCHMARK")
    print("=" * 70)

    # Load data
    X, y, sessions = load_data(data_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to test
    models = [
        ('gb', ModelConfig(model_type='gb', learning_rate=0.1, max_depth=5)),
        ('rf', ModelConfig(model_type='rf', n_estimators=100, max_depth=10)),
        ('logistic', ModelConfig(model_type='logistic', C=1.0)),
    ]

    print(f"\n[Benchmark Results]")
    print("-" * 70)
    print(f"{'Model':<15} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)

    results = []
    for name, config in models:
        try:
            model = ModelFactory.create(config)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            metrics = compute_metrics(y_test, y_pred)

            print(f"{name:<15} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")

            results.append((name, metrics))
        except Exception as e:
            print(f"{name:<15} Error: {e}")

    print("-" * 70)

    # Best model
    if results:
        best = max(results, key=lambda x: x[1]['f1'])
        print(f"\nBest model (by F1): {best[0]} (F1={best[1]['f1']:.4f})")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPG Fist Classifier")

    parser.add_argument("--data", type=str, required=True,
                        help="Path to NPZ data file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (for evaluation)")

    # LOSO mode
    parser.add_argument("--loso", action="store_true",
                        help="Run Leave-One-Session-Out cross-validation")
    parser.add_argument("--model-type", type=str, default='gb',
                        choices=['gb', 'rf', 'xgb', 'lgb', 'logistic'],
                        help="Model type for LOSO (default: gb)")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=5)

    # Benchmark mode
    parser.add_argument("--benchmark", action="store_true",
                        help="Run quick benchmark of multiple models")

    # Other options
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")

    args = parser.parse_args()

    if args.benchmark:
        quick_benchmark(args.data)
    elif args.loso:
        evaluate_loso(args.data, args.model_type,
                      args.learning_rate, args.max_depth)
    elif args.model:
        evaluate_model(args.model, args.data, args.test_size)
    else:
        print("Please specify --model, --loso, or --benchmark")
        print("\nExamples:")
        print("  python evaluate.py --data ./data/baseline_all.npz --benchmark")
        print("  python evaluate.py --data ./data/baseline_all.npz --loso")
        print("  python evaluate.py --data ./data/baseline_all.npz --model ./production_models/final_model_gb.pkl")


if __name__ == "__main__":
    main()
