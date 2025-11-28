# PPG-Based Fist Classifier

A machine learning system for real-time fist detection using PPG (Photoplethysmography) signals.

## Overview

- **Purpose**: Classify hand state (open/stable vs closed fist) from 16-channel PPG sensor time-series data
- **Language**: Python 3.12+
- **Sampling Rate**: 25 Hz
- **Evaluation**: LOSO (Leave-One-Session-Out) cross-validation

## Performance Summary

| Scenario | F1 Score | AUC | Description |
|----------|----------|-----|-------------|
| Zero-shot (LOSO) | 0.46 | 0.68 | Generic model for new users |
| Few-shot (20s calibration) | 0.73 | 0.88 | After 20s user calibration |
| User-dependent | 0.65 | 0.94 | User-specific model |

## Data

The training dataset is available on Google Drive:
- **[Download Dataset](https://drive.google.com/drive/folders/13Jly9BetyXIt-W287WnxIfIVWaeSLB2m?usp=sharing)**

Place the downloaded files in the appropriate directories (`data/`, `ppg_runs/`).

## Directory Structure

```
Final/
├── preprocessor.py              # Preprocessing & feature extraction
├── model.py                     # Model definitions
├── train.py                     # Model training
├── evaluate.py                  # Performance evaluation
├── realtime.py                  # Real-time inference
├── final_report_experiment.py   # Report experiment code
├── data/
│   └── baseline_all.npz         # Training dataset (download from Drive)
├── figures/                     # Experiment result figures
│   ├── fig1_loso_session_performance.png
│   ├── fig2_loso_roc_curve.png
│   ├── fig3_loso_confusion_matrix.png
│   ├── fig4_within_session_performance.png
│   ├── fig5_calibration_curve.png
│   ├── fig6_scenario_comparison.png
│   └── results_table.csv
├── final_models/
│   └── final_model_gb.pkl       # Trained model
├── production_models/
│   └── final_model_gb.pkl       # Production model
└── ppg_runs/                    # Raw PPG data
    ├── samples_*.csv
    └── tight/
```

## Core Files

| File | Purpose | Description |
|------|---------|-------------|
| `preprocessor.py` | Preprocessing | CSV → NPZ conversion, feature extraction (14 per channel) |
| `model.py` | Model | ModelConfig, ModelFactory, ModelPackage definitions |
| `train.py` | Training | Data loading → training → model saving |
| `evaluate.py` | Evaluation | Train/Test split, LOSO cross-validation, benchmarks |
| `realtime.py` | Real-time | UDP reception → prediction → calibration |
| `final_report_experiment.py` | Experiments | Generate report figures and tables |

## Usage

### 1. Feature Extraction
```bash
python preprocessor.py \
    --runs-glob "./ppg_runs/samples_*.csv" \
    --out ./data/baseline_all.npz \
    --fs 25.0 \
    --window-sec 3.0 \
    --stride-sec 0.5
```

### 2. Model Training
```bash
python train.py \
    --data ./data/baseline_all.npz \
    --model gb \
    --output-dir ./production_models
```

### 3. Performance Evaluation
```bash
# LOSO cross-validation
python evaluate.py \
    --data ./data/baseline_all.npz \
    --loso \
    --model-type gb
```

### 4. Real-time Inference
```bash
python realtime.py \
    --model ./production_models/final_model_gb.pkl \
    --port 65002
```

### 5. Run Report Experiments
```bash
python final_report_experiment.py
# Results saved to figures/ folder
```

## Data Pipeline

### Preprocessing Steps
1. **Bandpass Filter**: 0.5-10 Hz (Butterworth, order 4)
2. **Windowing**: 3-second window, 0.5-second stride
3. **Label Filtering**: pos_min=0.9, neg_max=0.1
4. **Transition Guard**: ±1.0 second (exclude label transition regions)

### Feature Extraction (14 per channel = 224 total)
- **Time-domain (6)**: mean, std, min, max, ptp, rms
- **Gradient (4)**: g_mean, g_std, g_rms, g_abs_mean
- **Frequency (3)**: band_power (0.5-2.5, 2.5-5, 5-10 Hz)
- **DC shift (1)**: relative_dc_shift

## Model Architecture

### Gradient Boosting (Traditional ML)
```python
HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=5,
    max_iter=200
)
```

### Multi-Scale CNN (Deep Learning)
```python
MultiScaleCNN:
    - Multi-scale convolutions (kernel 3, 5, 7)
    - BatchNorm + ReLU + MaxPool
    - Conv layers: 96 -> 128 -> 256
    - AdaptiveAvgPool + FC
```

## Key Findings

1. **Effective Channels**: ch01, ch05, ch07 (Cohen's d > 0.5)
2. **Optimal Window**: 3.0 seconds
3. **Session Domain Shift**: ~0.2 F1 drop in LOSO evaluation
4. **Calibration Effect**: 20s calibration improves F1 from 0.46 to 0.73

## Requirements

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
joblib>=1.0.0
torch>=1.10.0
matplotlib>=3.4.0
```

## UDP Data Format

- **Port**: 65002
- **Packet Size**: 80 bytes
- **Structure**: Header (4 floats: h,m,s,ns) + 16 int32 channel values

## License

MIT License
