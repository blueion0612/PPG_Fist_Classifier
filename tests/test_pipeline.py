"""Tests for the preprocessing chain and the model package.

Every test builds its own synthetic signal, so nothing here needs the
recordings. Run with `pytest`, or directly with
`python tests/test_pipeline.py` if pytest is not installed.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppg_fist_classifier.preprocessor import (  # noqa: E402
    PreprocessConfig,
    FeatureExtractor,
    design_bandpass,
    apply_bandpass,
    compute_bandpower,
)
from ppg_fist_classifier.model import ModelConfig, ModelFactory, ModelPackage  # noqa: E402

FS = 25.0


def _tone(freq, seconds=8.0, fs=FS, amp=1.0):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def test_config_derives_sample_counts():
    """Window, stride and guard sizes follow from the durations and rate."""
    c = PreprocessConfig(fs=25.0, window_sec=3.0, stride_sec=0.5, guard_sec=1.0)
    assert c.window_size == 75
    assert c.stride_size == 12
    assert c.guard_size == 25


def test_bandpass_rejects_out_of_band_content():
    """A 0.05 Hz drift and a 12 Hz tone are attenuated, a 2 Hz tone is not."""
    b, a = design_bandpass(FS, 0.5, 10.0, order=4)
    passband = apply_bandpass(_tone(2.0), b, a)
    drift = apply_bandpass(_tone(0.05), b, a)
    above = apply_bandpass(_tone(12.0), b, a)

    core = slice(50, -50)  # ignore filter edge transients
    assert passband[core].std() > 0.7
    assert drift[core].std() < 0.1
    assert above[core].std() < 0.1


def test_bandpass_is_zero_phase():
    """Filtering both ways leaves no group delay.

    A causal filter would move the peak of a symmetric pulse to the right by
    its group delay. Here it must stay exactly at the center. Residual
    asymmetry comes from edge padding and is bounded well below the peak.
    """
    b, a = design_bandpass(FS, 0.5, 10.0, order=4)
    n = 201
    x = np.zeros(n)
    x[n // 2] = 1.0                      # symmetric about its center
    y = apply_bandpass(x, b, a)

    assert int(np.argmax(np.abs(y))) == n // 2
    assert np.max(np.abs(y - y[::-1])) / np.max(np.abs(y)) < 1e-3


def test_bandpower_lands_in_the_right_band():
    """A pure 1.5 Hz tone puts its power in the 0.5-2.5 Hz band."""
    bands = FeatureExtractor.BANDS
    low, mid, high = compute_bandpower(_tone(1.5), FS, bands)
    assert low > mid and low > high

    low2, mid2, high2 = compute_bandpower(_tone(7.0), FS, bands)
    assert high2 > low2 and high2 > mid2


def test_feature_vector_length_matches_the_name_list():
    """One channel yields exactly 14 features, and the names agree."""
    x = _tone(1.5)
    feats = FeatureExtractor.extract_channel_features(x, x + 1000.0, FS, 1000.0)
    assert len(feats) == FeatureExtractor.FEATURES_PER_CHANNEL == 14
    assert all(np.isfinite(feats))

    names = FeatureExtractor.get_feature_names([f"ch{i:02d}" for i in range(16)])
    assert len(names) == 16 * 14
    assert len(set(names)) == len(names)
    assert names[:2] == ["ch00_mean", "ch00_std"]


def test_relative_dc_shift_tracks_the_baseline():
    """The DC feature is the fractional change against the resting level."""
    x = np.zeros(75)
    feats = FeatureExtractor.extract_channel_features(x, x + 1100.0, FS, 1000.0)
    assert abs(feats[-1] - 0.1) < 1e-3


def test_factory_builds_every_advertised_model():
    """available_models() does not advertise anything the factory cannot build."""
    for name in ModelFactory.available_models():
        est = ModelFactory.create(ModelConfig(model_type=name))
        assert hasattr(est, "fit") and hasattr(est, "predict")


def test_model_package_round_trips():
    """A saved package reloads and predicts identically."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 14))
    y = (X[:, 0] > 0).astype(int)

    scaler = StandardScaler().fit(X)
    cfg = ModelConfig(model_type="gb", max_iter=20)
    est = ModelFactory.create(cfg).fit(scaler.transform(X), y)
    pkg = ModelPackage(model=est, scaler=scaler, config=cfg)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pkl")
        pkg.save(path)
        again = ModelPackage.load(path)

    assert np.array_equal(pkg.predict(X), again.predict(X))
    assert again.config.model_type == "gb"
    assert again.config.max_iter == 20


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
