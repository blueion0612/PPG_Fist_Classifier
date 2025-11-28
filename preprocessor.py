#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocessor.py
---------------
PPG 신호 전처리 및 특징 추출 모듈

사용법:
    python preprocessor.py \
        --runs-glob "./ppg_runs/samples_*.csv" \
        --out ./data/baseline_all.npz \
        --fs 25.0 --window-sec 3.0 --stride-sec 0.5 \
        --pos-min 0.9 --neg-max 0.1 --guard-sec 1.0
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PreprocessConfig:
    """전처리 설정"""
    fs: float = 25.0              # 샘플링 레이트 (Hz)
    window_sec: float = 3.0       # 윈도우 길이 (초)
    stride_sec: float = 0.5       # stride (초)
    pos_min: float = 0.9          # positive 레이블 최소 비율
    neg_max: float = 0.1          # negative 레이블 최대 비율
    guard_sec: float = 1.0        # transition guard (초)
    bp_low: float = 0.5           # bandpass 하한 (Hz)
    bp_high: float = 10.0         # bandpass 상한 (Hz)
    bp_order: int = 4             # bandpass filter order

    @property
    def window_size(self) -> int:
        return int(self.window_sec * self.fs)

    @property
    def stride_size(self) -> int:
        return int(self.stride_sec * self.fs)

    @property
    def guard_size(self) -> int:
        return int(self.guard_sec * self.fs)


# =============================================================================
# Signal Processing
# =============================================================================

def design_bandpass(fs: float, low: float, high: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Butterworth bandpass filter 설계"""
    nyq = 0.5 * fs
    low_norm = max(0.01, low / nyq)
    high_norm = min(0.99, high / nyq)
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return b, a


def apply_bandpass(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Zero-phase bandpass filter 적용"""
    return filtfilt(b, a, x, axis=0)


def compute_bandpower(signal: np.ndarray, fs: float, bands: List[Tuple[float, float]]) -> List[float]:
    """주파수 대역별 power 계산"""
    L = len(signal)
    if L == 0:
        return [0.0] * len(bands)

    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    spec = np.fft.rfft(signal)
    psd = (np.abs(spec) ** 2) / L

    bp = []
    for (f_lo, f_hi) in bands:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        bp.append(float(psd[mask].sum()) if np.any(mask) else 0.0)

    return bp


# =============================================================================
# Feature Extraction
# =============================================================================

class FeatureExtractor:
    """채널당 14개 특징 추출"""

    BANDS = [(0.5, 2.5), (2.5, 5.0), (5.0, 10.0)]
    FEATURES_PER_CHANNEL = 14

    @staticmethod
    def extract_channel_features(x: np.ndarray, x_raw: np.ndarray,
                                  fs: float, baseline_raw_mean: float) -> List[float]:
        """단일 채널에서 14개 특징 추출

        Features:
            Time-domain (6): mean, std, min, max, ptp, rms
            Gradient (4): g_mean, g_std, g_rms, g_abs_mean
            Frequency (3): band_power (3 bands)
            DC shift (1): relative_dc_shift
        """
        features = []

        # Time-domain features (6)
        features.extend([
            float(x.mean()),
            float(x.std()),
            float(x.min()),
            float(x.max()),
            float(x.max() - x.min()),  # ptp
            float(np.sqrt(np.mean(x**2))),  # rms
        ])

        # Gradient features (4)
        dx = np.diff(x)
        if len(dx) > 0:
            features.extend([
                float(dx.mean()),
                float(dx.std()),
                float(np.sqrt(np.mean(dx**2))),  # g_rms
                float(np.mean(np.abs(dx))),  # g_abs_mean
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Band power (3)
        bp_vals = compute_bandpower(x, fs, FeatureExtractor.BANDS)
        features.extend(bp_vals)

        # Relative DC shift (1)
        raw_mean = float(x_raw.mean())
        rel_dc = (raw_mean - baseline_raw_mean) / (abs(baseline_raw_mean) + 1e-6)
        features.append(rel_dc)

        return features

    @staticmethod
    def get_feature_names(channel_names: List[str]) -> List[str]:
        """특징 이름 생성"""
        feature_suffixes = [
            "mean", "std", "min", "max", "ptp", "rms",
            "g_mean", "g_std", "g_rms", "g_abs_mean",
            "bp_0_2p5", "bp_2p5_5", "bp_5_10",
            "rel_dc"
        ]

        names = []
        for ch in channel_names:
            for suffix in feature_suffixes:
                names.append(f"{ch}_{suffix}")
        return names


# =============================================================================
# Data Processing Pipeline
# =============================================================================

class PPGPreprocessor:
    """PPG 데이터 전처리 파이프라인"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.b, self.a = design_bandpass(
            config.fs, config.bp_low, config.bp_high, config.bp_order
        )

    def process_csv(self, csv_path: str) -> Optional[Dict]:
        """CSV 파일 처리"""
        # Load data
        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            raise ValueError(f"'label' column not found in {csv_path}")

        labels = df["label"].values.astype(int)

        # Get channel columns
        channels = [col for col in df.columns if col.startswith('ch')]
        if len(channels) == 0:
            channels = [col for col in df.columns
                       if col not in ['label', 't_abs_host']]

        data_raw = df[channels].values.astype(np.float32)

        # Filter valid labels (0,1,2,3)
        valid_mask = np.isin(labels, [0, 1, 2, 3])
        labels = labels[valid_mask]
        data_raw = data_raw[valid_mask]

        print(f"  Loaded {len(data_raw)} samples, {len(channels)} channels")

        # Apply bandpass filter
        data_filtered = apply_bandpass(data_raw, self.b, self.a)

        # Calibration using first 5 seconds of stable data
        calib_samples = min(int(5.0 * self.config.fs), len(labels))
        stable_mask = labels[:calib_samples] != 1

        if np.sum(stable_mask) < 10:
            stable_mask = np.ones(calib_samples, dtype=bool)

        # Compute baselines
        baseline_mean = data_filtered[:calib_samples][stable_mask].mean(axis=0)
        baseline_std = data_filtered[:calib_samples][stable_mask].std(axis=0)
        baseline_std[baseline_std < 1e-6] = 1.0
        baseline_raw_mean = data_raw[:calib_samples][stable_mask].mean(axis=0)

        # Normalize
        data_norm = (data_filtered - baseline_mean) / baseline_std

        # Create binary labels (1=fist, 0=stable)
        binary_labels = np.zeros_like(labels)
        binary_labels[labels == 1] = 1

        # Find transition points and create valid mask
        transitions = np.where(np.diff(binary_labels) != 0)[0]
        valid_mask = np.ones(len(labels), dtype=bool)

        for t in transitions:
            start = max(0, t - self.config.guard_size)
            end = min(len(labels), t + self.config.guard_size + 1)
            valid_mask[start:end] = False

        # Extract windows and features
        features = []
        labels_out = []

        for start_idx in range(0, len(labels) - self.config.window_size + 1,
                               self.config.stride_size):
            end_idx = start_idx + self.config.window_size

            # Skip invalid windows
            if not np.all(valid_mask[start_idx:end_idx]):
                continue

            # Get label ratio
            window_binary = binary_labels[start_idx:end_idx]
            label_ratio = np.mean(window_binary)

            # Determine window label
            if label_ratio >= self.config.pos_min:
                y_win = 1  # fist
            elif label_ratio <= self.config.neg_max:
                y_win = 0  # stable
            else:
                continue  # skip ambiguous

            # Extract features
            window_norm = data_norm[start_idx:end_idx]
            window_raw = data_raw[start_idx:end_idx]

            feat_vec = []
            for ch_i in range(len(channels)):
                ch_features = FeatureExtractor.extract_channel_features(
                    window_norm[:, ch_i],
                    window_raw[:, ch_i],
                    self.config.fs,
                    baseline_raw_mean[ch_i]
                )
                feat_vec.extend(ch_features)

            features.append(feat_vec)
            labels_out.append(y_win)

        if len(features) == 0:
            return None

        session_name = os.path.splitext(os.path.basename(csv_path))[0]

        return {
            "X": np.asarray(features, dtype=np.float32),
            "y": np.asarray(labels_out, dtype=np.int64),
            "session": np.array([session_name] * len(labels_out)),
            "feature_names": FeatureExtractor.get_feature_names(channels),
            "n_channels": len(channels)
        }

    def process_multiple(self, csv_paths: List[str]) -> Dict:
        """여러 CSV 파일 처리 및 병합"""
        all_X, all_y, all_sessions = [], [], []
        feature_names = None

        for csv_path in csv_paths:
            print(f"\nProcessing: {csv_path}")
            result = self.process_csv(csv_path)

            if result is not None:
                all_X.append(result["X"])
                all_y.append(result["y"])
                all_sessions.append(result["session"])
                if feature_names is None:
                    feature_names = result["feature_names"]
                print(f"  Extracted {len(result['y'])} windows")

        X = np.vstack(all_X) if all_X else np.empty((0, 0))
        y = np.hstack(all_y) if all_y else np.empty(0)
        sessions = np.hstack(all_sessions) if all_sessions else np.empty(0)

        return {
            "X": X,
            "y": y,
            "sessions": sessions,
            "feature_names": feature_names
        }

    def save(self, data: Dict, output_path: str):
        """NPZ 파일로 저장"""
        meta = {
            "fs": self.config.fs,
            "window_sec": self.config.window_sec,
            "stride_sec": self.config.stride_sec,
            "guard_sec": self.config.guard_sec,
            "bp": [self.config.bp_low, self.config.bp_high],
            "pos_min": self.config.pos_min,
            "neg_max": self.config.neg_max,
            "feature_names": data["feature_names"]
        }

        np.savez(
            output_path,
            X=data["X"],
            y=data["y"],
            session=data["sessions"],
            meta=np.array(meta, dtype=object)
        )

        print(f"\nSaved to: {output_path}")
        print(f"  Total windows: {len(data['y'])}")
        print(f"  Features: {data['X'].shape[1]}")
        print(f"  Class distribution: 0={np.sum(data['y']==0)}, 1={np.sum(data['y']==1)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPG Feature Extraction")
    parser.add_argument("--runs-glob", type=str, required=True,
                        help="Glob pattern for CSV files")
    parser.add_argument("--out", type=str, required=True,
                        help="Output NPZ file path")
    parser.add_argument("--fs", type=float, default=25.0,
                        help="Sampling rate (Hz)")
    parser.add_argument("--window-sec", type=float, default=3.0,
                        help="Window length (seconds)")
    parser.add_argument("--stride-sec", type=float, default=0.5,
                        help="Stride length (seconds)")
    parser.add_argument("--pos-min", type=float, default=0.9,
                        help="Minimum ratio for positive label")
    parser.add_argument("--neg-max", type=float, default=0.1,
                        help="Maximum ratio for negative label")
    parser.add_argument("--guard-sec", type=float, default=1.0,
                        help="Transition guard (seconds)")
    parser.add_argument("--bp-low", type=float, default=0.5,
                        help="Bandpass low cutoff (Hz)")
    parser.add_argument("--bp-high", type=float, default=10.0,
                        help="Bandpass high cutoff (Hz)")

    args = parser.parse_args()

    # Create config
    config = PreprocessConfig(
        fs=args.fs,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        pos_min=args.pos_min,
        neg_max=args.neg_max,
        guard_sec=args.guard_sec,
        bp_low=args.bp_low,
        bp_high=args.bp_high
    )

    # Find CSV files
    csv_files = sorted(glob.glob(args.runs_glob))
    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        print(f"Error: No files found with pattern: {args.runs_glob}")
        return

    # Process
    preprocessor = PPGPreprocessor(config)
    data = preprocessor.process_multiple(csv_files)

    # Save
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    preprocessor.save(data, args.out)


if __name__ == "__main__":
    main()
