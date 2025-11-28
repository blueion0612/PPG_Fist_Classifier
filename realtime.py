#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
realtime.py
-----------
PPG 실시간 추론 시스템

기능:
    1. UDP 스트림에서 PPG 데이터 수신
    2. 실시간 특징 추출 및 예측
    3. 사용자 캘리브레이션 지원
    4. Temporal Smoothing

사용법:
    python realtime.py --model ./production_models/final_model_gb.pkl
"""

import argparse
import numpy as np
from collections import deque
import time
import threading
import socket
import queue
import sys
import os
import warnings

warnings.filterwarnings('ignore')

from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from model import ModelPackage
from preprocessor import FeatureExtractor

# Windows keyboard input
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


# =============================================================================
# UDP Listener
# =============================================================================

class PPGUdpListener(threading.Thread):
    """UDP 스트림 리스너"""

    def __init__(self, bind_ip: str = "0.0.0.0", port: int = 65002, endian: str = "big"):
        super().__init__(daemon=True)
        self.bind_ip = bind_ip
        self.port = port

        byte_order = "<" if endian == "little" else ">"
        self.f4 = np.dtype(byte_order + "f4")
        self.i4 = np.dtype(byte_order + "i4")

        self.sock = None
        self.q = queue.Queue(maxsize=2000)
        self._running = threading.Event()
        self._running.set()

    def stop(self):
        self._running.clear()
        if self.sock:
            self.sock.close()

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.port))
        self.sock.settimeout(0.5)

        print(f"[UDP] Listening on {self.bind_ip}:{self.port}")

        while self._running.is_set():
            try:
                data, addr = self.sock.recvfrom(80)
                t_host = time.time()
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) != 80:
                continue

            try:
                # Parse packet: Header (4 floats) + Channels (16 int32)
                chi = np.frombuffer(data, dtype=self.i4, count=16, offset=16)
                values = chi.astype(np.float32)

                try:
                    self.q.put_nowait((t_host, values))
                except queue.Full:
                    try:
                        self.q.get_nowait()
                        self.q.put_nowait((t_host, values))
                    except:
                        pass
            except Exception as e:
                continue


# =============================================================================
# Realtime Feature Extractor
# =============================================================================

class RealtimeFeatureExtractor:
    """실시간 특징 추출기"""

    def __init__(self, fs: float = 25.0, window_sec: float = 3.0, stride_sec: float = 0.5):
        self.fs = fs
        self.window_size = int(window_sec * fs)
        self.stride = int(stride_sec * fs)

        self.buffer = []
        self.sample_count = 0

        # Calibration
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_raw_mean = None
        self.calibrated = False

    def add_sample(self, values: np.ndarray):
        """샘플 추가"""
        self.buffer.append(values)
        self.sample_count += 1

        # Limit buffer size
        if len(self.buffer) > self.window_size * 3:
            self.buffer = self.buffer[-self.window_size * 2:]

    def can_extract(self) -> bool:
        """특징 추출 가능 여부"""
        if len(self.buffer) < self.window_size:
            return False
        return self.sample_count % self.stride == 0

    def calibrate(self, stable_samples: list) -> bool:
        """안정 상태 데이터로 캘리브레이션"""
        if len(stable_samples) < 10:
            return False

        data = np.array(stable_samples)
        self.baseline_mean = np.mean(data, axis=0)
        self.baseline_std = np.std(data, axis=0)
        self.baseline_std[self.baseline_std < 1e-6] = 1.0
        self.baseline_raw_mean = self.baseline_mean.copy()
        self.calibrated = True

        print(f"[Calibrated with {len(stable_samples)} samples]")
        return True

    def extract_features(self) -> np.ndarray:
        """현재 윈도우에서 특징 추출"""
        if len(self.buffer) < self.window_size:
            return None

        window = np.array(self.buffer[-self.window_size:])

        # Normalize
        if self.calibrated:
            window_norm = (window - self.baseline_mean) / self.baseline_std
        else:
            window_norm = window

        # Extract features (14 per channel)
        features = []
        for ch in range(16):
            ch_features = FeatureExtractor.extract_channel_features(
                window_norm[:, ch],
                window[:, ch],
                self.fs,
                self.baseline_raw_mean[ch] if self.calibrated else 0.0
            )
            features.extend(ch_features)

        return np.array(features)


# =============================================================================
# Calibration Manager
# =============================================================================

class CalibrationManager:
    """사용자 캘리브레이션 관리"""

    def __init__(self, base_model, scaler: StandardScaler):
        self.base_model = base_model
        self.scaler = scaler

        self.features = []
        self.labels = []

        self.current_label = 0
        self.switch_time = time.time()
        self.switch_interval = 5.0

        self.personal_model = None
        self.alpha = 0.5
        self.active = True

    def update(self) -> bool:
        """레이블 업데이트 (시간 기반)"""
        if time.time() - self.switch_time >= self.switch_interval:
            self.current_label = 1 - self.current_label
            self.switch_time = time.time()
            return True
        return False

    def get_remaining_time(self) -> float:
        """다음 전환까지 남은 시간"""
        return max(0, self.switch_interval - (time.time() - self.switch_time))

    def add_sample(self, features: np.ndarray):
        """캘리브레이션 샘플 추가"""
        if self.active and features is not None:
            self.features.append(features)
            self.labels.append(self.current_label)

    def train_personal_model(self) -> bool:
        """개인화 모델 학습"""
        n_stable = self.labels.count(0)
        n_fist = self.labels.count(1)

        if len(self.features) < 20 or n_stable < 5 or n_fist < 5:
            return False

        X = np.array(self.features)
        y = np.array(self.labels)
        X_scaled = self.scaler.transform(X)

        self.personal_model = LogisticRegression(
            C=0.5, class_weight='balanced', max_iter=500, random_state=42
        )

        try:
            self.personal_model.fit(X_scaled, y)
            y_pred = self.personal_model.predict(X_scaled)
            f1 = f1_score(y, y_pred)

            # Adjust alpha based on performance
            self.alpha = 0.7 if f1 > 0.8 else (0.5 if f1 > 0.6 else 0.3)

            print(f"[Personal model trained: F1={f1:.3f}, alpha={self.alpha}]")
            return True
        except:
            return False

    def predict(self, features: np.ndarray) -> int:
        """앙상블 예측"""
        if features is None:
            return None

        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        if self.personal_model is None:
            return int(self.base_model.predict(X_scaled)[0])

        try:
            p_base = self.base_model.predict_proba(X_scaled)[0, 1]
            p_personal = self.personal_model.predict_proba(X_scaled)[0, 1]
            p_ensemble = self.alpha * p_personal + (1 - self.alpha) * p_base
            return int(p_ensemble >= 0.5)
        except:
            return int(self.base_model.predict(X_scaled)[0])

    def get_stats(self) -> dict:
        """통계 정보"""
        return {
            'n_samples': len(self.labels),
            'n_stable': self.labels.count(0),
            'n_fist': self.labels.count(1),
            'has_personal': self.personal_model is not None,
            'alpha': self.alpha
        }


# =============================================================================
# Realtime Inference System
# =============================================================================

class RealtimeInference:
    """실시간 추론 시스템"""

    def __init__(self, model_path: str, bind_ip: str = "0.0.0.0", port: int = 65002):
        # Load model
        print("\n[Loading model...]")
        self.package = ModelPackage.load(model_path)
        print(f"  Type: {self.package.config.model_type}")
        print(f"  Features: {self.package.training_info.get('n_features', 'N/A')}")

        # Components
        self.listener = PPGUdpListener(bind_ip, port)
        self.extractor = RealtimeFeatureExtractor()
        self.calibration = CalibrationManager(self.package.model, self.package.scaler)

        # Prediction history
        self.predictions = deque(maxlen=10)
        self.use_smoothing = self.package.config.use_smoothing

        # Stats
        self.pps = 0
        self.pps_counter = 0
        self.pps_time = time.time()

        # Control
        self._running = True
        self.inference_mode = False

    def start(self):
        """시스템 시작"""
        self.listener.start()

        # Start display thread
        threading.Thread(target=self._display_loop, daemon=True).start()

        # Start key input thread
        threading.Thread(target=self._key_loop, daemon=True).start()

        # Main loop
        self._main_loop()

    def _display_loop(self):
        """화면 출력"""
        while self._running:
            os.system('cls' if os.name == 'nt' else 'clear')

            print("=" * 60)
            print("PPG REAL-TIME FIST CLASSIFIER")
            print("=" * 60)

            print(f"\n[Network] {self.listener.bind_ip}:{self.listener.port}")
            print(f"  PPS: {self.pps}")
            print(f"  Buffer: {len(self.extractor.buffer)}/{self.extractor.window_size}")
            print(f"  Calibrated: {self.extractor.calibrated}")

            stats = self.calibration.get_stats()

            if self.calibration.active:
                print(f"\n[CALIBRATION MODE]")
                label_text = "RELAX HAND" if self.calibration.current_label == 0 else "MAKE FIST"
                print(f"  Instruction: {label_text}")
                print(f"  Next switch in: {self.calibration.get_remaining_time():.1f}s")
                print(f"\n  Samples: {stats['n_samples']} (Stable: {stats['n_stable']}, Fist: {stats['n_fist']})")

                if stats['has_personal']:
                    print(f"  Personal model: Trained (alpha={stats['alpha']:.2f})")
            else:
                print(f"\n[INFERENCE MODE]")
                if len(self.predictions) > 0:
                    pred = self.predictions[-1]
                    pred_text = "FIST" if pred == 1 else "STABLE"
                    print(f"  Current: {pred_text}")

                    if len(self.predictions) >= 5:
                        history = list(self.predictions)[-5:]
                        print(f"  Recent: {history}")

            print("\n" + "-" * 60)
            print("Controls:")
            if self.calibration.active:
                print("  [SPACE] End calibration, start inference")
                print("  [R]     Reset calibration")
            else:
                print("  [C]     Return to calibration")
            print("  [Q]     Quit")
            print("-" * 60)

            time.sleep(0.5)

    def _key_loop(self):
        """키보드 입력"""
        if HAS_MSVCRT:
            while self._running:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch().lower()
                    self._handle_key(ch)
                time.sleep(0.05)
        else:
            import select
            while self._running:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1).lower()
                    self._handle_key(ch)
                time.sleep(0.05)

    def _handle_key(self, ch: str):
        """키 입력 처리"""
        if ch == ' ':
            if self.calibration.active:
                self.calibration.active = False
                self.inference_mode = True
                print("\n[Starting inference mode]")

        elif ch == 'c':
            self.calibration.active = True
            self.inference_mode = False

        elif ch == 'r':
            self.calibration.features.clear()
            self.calibration.labels.clear()
            self.calibration.personal_model = None
            self.calibration.switch_time = time.time()
            self.calibration.current_label = 0
            print("[Calibration reset]")

        elif ch == 'q':
            self._running = False

    def _main_loop(self):
        """메인 처리 루프"""
        print("\n[System started. Waiting for data...]")

        initial_buffer = []
        initial_done = False

        while self._running:
            try:
                # Get UDP data
                try:
                    t_host, values = self.listener.q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Update PPS
                self.pps_counter += 1
                if time.time() - self.pps_time >= 1.0:
                    self.pps = self.pps_counter
                    self.pps_counter = 0
                    self.pps_time = time.time()

                # Add to buffer
                self.extractor.add_sample(values)

                # Initial calibration
                if not self.extractor.calibrated:
                    initial_buffer.append(values)
                    if len(initial_buffer) >= 50:  # 2 seconds
                        self.extractor.calibrate(initial_buffer[:50])
                        initial_done = True
                        self.calibration.switch_time = time.time()
                    continue

                # Feature extraction
                if self.extractor.can_extract():
                    features = self.extractor.extract_features()

                    if features is not None:
                        if self.calibration.active:
                            # Calibration mode
                            if self.calibration.update():
                                instruction = "RELAX HAND" if self.calibration.current_label == 0 else "MAKE FIST"
                                print(f"\n[Switch! {instruction}]")

                            self.calibration.add_sample(features)

                            if len(self.calibration.features) % 10 == 0:
                                self.calibration.train_personal_model()

                        elif self.inference_mode:
                            # Inference mode
                            prediction = self.calibration.predict(features)

                            if prediction is not None:
                                self.predictions.append(prediction)

                                # Apply smoothing
                                if self.use_smoothing and len(self.predictions) >= 3:
                                    smoothed = median_filter(
                                        list(self.predictions)[-5:],
                                        size=min(3, len(list(self.predictions)[-5:]))
                                    )
                                    final_pred = int(smoothed[-1])
                                else:
                                    final_pred = prediction

            except Exception as e:
                print(f"[Error: {e}]")
                continue

        # Cleanup
        self.listener.stop()
        print("\n[System shutdown]")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPG Realtime Inference")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--ip", type=str, default="0.0.0.0",
                        help="Bind IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=65002,
                        help="UDP port (default: 65002)")
    parser.add_argument("--endian", choices=["little", "big"], default="big",
                        help="Byte order (default: big)")

    args = parser.parse_args()

    system = RealtimeInference(args.model, args.ip, args.port)

    try:
        system.start()
    except KeyboardInterrupt:
        print("\n[Interrupted]")


if __name__ == "__main__":
    main()
