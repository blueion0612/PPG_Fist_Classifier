"""The numbers the README states are the numbers docs/figures/results.json holds.

results.json is written by scripts/report_experiment.py. The README table and the
figures both read from it, so this test fails the moment the README drifts.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# README row label -> key in results.json
ROWS = {
    "Leave-one-session-out, no calibration": "LOSO (Zero-shot)",
    "Calibrated on the first 10 s of the held-out session": "Few-shot (10s)",
    "Calibrated on the first 20 s": "Few-shot (20s)",
    "Calibrated on the first 30 s": "Few-shot (30s)",
    "Trained and tested within one session": "Within-Session (User-dep)",
    "Leave-one-session-out with per-session baseline subtraction": "LOSO + Baseline Sub",
}


def _load():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    with open(os.path.join(ROOT, "docs", "figures", "results.json"), encoding="utf-8") as fh:
        results = json.load(fh)
    return readme, results


def test_results_table_matches_results_json():
    readme, results = _load()
    for label, key in ROWS.items():
        pattern = r"^\| " + re.escape(label) + r" \| \**([0-9.]+)\** \| \**([0-9.]+)\** \|$"
        row = re.search(pattern, readme, re.M)
        assert row, f"README has no results row for {label!r}"
        f1, auc = float(row.group(1)), float(row.group(2))
        assert f1 == round(results["scenarios"][key]["f1"], 2), key
        assert auc == round(results["scenarios"][key]["auc"], 2), key


def test_session_spread_matches_results_json():
    readme, results = _load()
    vals = list(results["loso_session_f1"].values())
    assert f"{min(vals):.2f} to {max(vals):.2f}" in readme
    assert f"mean of {results['loso_mean_f1']:.3f}" in readme
