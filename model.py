#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model.py
--------
PPG Fist Classifier 모델 정의

지원 모델:
    - gb: HistGradientBoostingClassifier (기본, 최고 성능)
    - rf: RandomForestClassifier
    - xgb: XGBClassifier (선택적)
    - lgb: LGBMClassifier (선택적)
    - logistic: LogisticRegression
"""

import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """모델 설정"""
    model_type: str = 'gb'

    # Gradient Boosting
    learning_rate: float = 0.1
    max_depth: int = 5
    max_iter: int = 200
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0

    # Random Forest
    n_estimators: int = 300
    min_samples_split: int = 5
    max_features: str = 'sqrt'

    # XGBoost/LightGBM
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    num_leaves: int = 31
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8

    # Logistic Regression
    C: float = 1.0
    penalty: str = 'l2'

    # Smoothing (for inference)
    use_smoothing: bool = True
    smooth_min_duration: int = 3
    smooth_window_size: int = 3

    # Ensemble
    ensemble_alpha: float = 0.5


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """모델 생성 팩토리"""

    @staticmethod
    def create(config: ModelConfig):
        """설정에 따라 모델 생성"""
        model_type = config.model_type

        if model_type == 'gb':
            return HistGradientBoostingClassifier(
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                max_iter=config.max_iter,
                min_samples_leaf=config.min_samples_leaf,
                l2_regularization=config.l2_regularization,
                random_state=42,
                verbose=0
            )

        elif model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )

        elif model_type == 'xgb':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )

        elif model_type == 'lgb':
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
            return lgb.LGBMClassifier(
                n_estimators=config.n_estimators,
                num_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                feature_fraction=config.feature_fraction,
                bagging_fraction=config.bagging_fraction,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            )

        elif model_type == 'logistic':
            solver = 'liblinear' if config.penalty == 'l1' else 'lbfgs'
            return LogisticRegression(
                C=config.C,
                penalty=config.penalty,
                solver=solver,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def available_models() -> List[str]:
        """사용 가능한 모델 목록"""
        models = ['gb', 'rf', 'logistic']
        if HAS_XGBOOST:
            models.append('xgb')
        if HAS_LIGHTGBM:
            models.append('lgb')
        return models


# =============================================================================
# Model Package
# =============================================================================

@dataclass
class ModelPackage:
    """학습된 모델 패키지"""
    model: Any
    scaler: StandardScaler
    config: ModelConfig
    training_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str):
        """모델 저장"""
        package = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.config.model_type,
            'model_params': {
                'learning_rate': self.config.learning_rate,
                'max_depth': self.config.max_depth,
                'max_iter': self.config.max_iter,
                'min_samples_leaf': self.config.min_samples_leaf,
                'n_estimators': self.config.n_estimators,
            },
            'training_info': self.training_info,
            'smoothing_params': {
                'use_smoothing': self.config.use_smoothing,
                'smooth_min_duration': self.config.smooth_min_duration,
                'smooth_window_size': self.config.smooth_window_size,
            },
            'ensemble_params': {
                'ensemble_alpha': self.config.ensemble_alpha,
            },
            'metadata': self.metadata
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(package, path)

        # Save config text file
        config_path = output_path.with_suffix('.txt')
        self._save_config(config_path)

        print(f"Model saved to: {path}")

    def _save_config(self, path: Path):
        """설정 파일 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("MODEL CONFIGURATION\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Training Samples: {self.training_info.get('n_samples', 'N/A')}\n")
            f.write(f"Features: {self.training_info.get('n_features', 'N/A')}\n")
            f.write(f"Training Accuracy: {self.training_info.get('train_accuracy', 0):.4f}\n")
            f.write(f"Training Time: {self.training_info.get('train_time', 0):.1f}s\n\n")

            f.write("Model Parameters:\n")
            f.write(f"  learning_rate: {self.config.learning_rate}\n")
            f.write(f"  max_depth: {self.config.max_depth}\n")
            f.write(f"  max_iter: {self.config.max_iter}\n")
            f.write(f"  min_samples_leaf: {self.config.min_samples_leaf}\n\n")

            f.write("Smoothing:\n")
            f.write(f"  Enabled: {self.config.use_smoothing}\n")
            f.write(f"  Min Duration: {self.config.smooth_min_duration}\n")
            f.write(f"  Window Size: {self.config.smooth_window_size}\n")

    @staticmethod
    def load(path: str) -> 'ModelPackage':
        """모델 로드"""
        data = joblib.load(path)

        config = ModelConfig(
            model_type=data.get('model_type', 'gb'),
            learning_rate=data.get('model_params', {}).get('learning_rate', 0.1),
            max_depth=data.get('model_params', {}).get('max_depth', 5),
            max_iter=data.get('model_params', {}).get('max_iter', 200),
            min_samples_leaf=data.get('model_params', {}).get('min_samples_leaf', 20),
            use_smoothing=data.get('smoothing_params', {}).get('use_smoothing', True),
            smooth_min_duration=data.get('smoothing_params', {}).get('smooth_min_duration', 3),
            smooth_window_size=data.get('smoothing_params', {}).get('smooth_window_size', 3),
            ensemble_alpha=data.get('ensemble_params', {}).get('ensemble_alpha', 0.5),
        )

        return ModelPackage(
            model=data['model'],
            scaler=data['scaler'],
            config=config,
            training_info=data.get('training_info', {}),
            metadata=data.get('metadata', {})
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Available models:", ModelFactory.available_models())

    # Test model creation
    config = ModelConfig(model_type='gb')
    model = ModelFactory.create(config)
    print(f"Created model: {type(model).__name__}")
