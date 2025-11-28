#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
--------
PPG Fist Classifier 모델 학습

사용법:
    python train.py \
        --data ./data/baseline_all.npz \
        --model gb \
        --learning-rate 0.1 \
        --max-depth 5 \
        --use-smoothing \
        --output-dir ./production_models
"""

import argparse
import numpy as np
import time
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler

from model import ModelConfig, ModelFactory, ModelPackage


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path: str):
    """NPZ 데이터 로드"""
    data = np.load(data_path, allow_pickle=True)

    X = data['X']
    y = data['y']
    sessions = data['session'] if 'session' in data else None
    meta = data['meta'].item() if 'meta' in data else {}

    print(f"\n[Data Loaded]")
    print(f"  Path: {data_path}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {dict(Counter(y))}")
    if sessions is not None:
        unique_sessions = np.unique(sessions)
        print(f"  Sessions: {list(unique_sessions)}")

    return X, y, sessions, meta


# =============================================================================
# Training
# =============================================================================

def train(args):
    """모델 학습"""
    print("\n" + "=" * 70)
    print("PPG FIST CLASSIFIER TRAINING")
    print("=" * 70)

    # Load data
    X, y, sessions, meta = load_data(args.data)

    # Create config
    config = ModelConfig(
        model_type=args.model,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_iter=args.max_iter,
        min_samples_leaf=args.min_samples_leaf,
        n_estimators=args.n_estimators,
        use_smoothing=args.use_smoothing,
        smooth_min_duration=args.smooth_min_duration,
        smooth_window_size=args.smooth_window_size,
        ensemble_alpha=args.ensemble_alpha,
    )

    print(f"\n[Model Configuration]")
    print(f"  Type: {config.model_type}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Max Depth: {config.max_depth}")
    print(f"  Max Iter: {config.max_iter}")
    print(f"  Min Samples Leaf: {config.min_samples_leaf}")

    # Create model
    model = ModelFactory.create(config)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    print(f"\n[Training]")
    start_time = time.time()
    model.fit(X_scaled, y)
    train_time = time.time() - start_time

    print(f"  Completed in {train_time:.1f} seconds")

    # Evaluate on training data
    y_pred = model.predict(X_scaled)
    train_acc = np.mean(y_pred == y)
    print(f"  Training Accuracy: {train_acc:.4f}")

    # Create model package
    training_info = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'train_time': train_time,
        'train_accuracy': float(train_acc),
        'classes': list(np.unique(y)),
        'class_distribution': dict(Counter(y))
    }

    package = ModelPackage(
        model=model,
        scaler=scaler,
        config=config,
        training_info=training_info,
        metadata=meta
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"final_model_{args.model}.pkl"
    package.save(str(model_path))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return package


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train PPG Fist Classifier")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to NPZ data file")
    parser.add_argument("--output-dir", type=str, default="./production_models",
                        help="Output directory")

    # Model selection
    parser.add_argument("--model", type=str, default='gb',
                        choices=['gb', 'rf', 'xgb', 'lgb', 'logistic'],
                        help="Model type (default: gb)")

    # Gradient Boosting parameters
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Max tree depth (default: 5)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max iterations (default: 200)")
    parser.add_argument("--min-samples-leaf", type=int, default=20,
                        help="Min samples in leaf (default: 20)")

    # Random Forest parameters
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees (default: 300)")

    # Smoothing
    parser.add_argument("--use-smoothing", action="store_true",
                        help="Enable temporal smoothing")
    parser.add_argument("--smooth-min-duration", type=int, default=3,
                        help="Min duration for smoothing (default: 3)")
    parser.add_argument("--smooth-window-size", type=int, default=3,
                        help="Smoothing window size (default: 3)")

    # Ensemble
    parser.add_argument("--ensemble-alpha", type=float, default=0.5,
                        help="Ensemble weight for personal model (default: 0.5)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
