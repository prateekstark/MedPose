"""End-to-end comparison against generated references (.mot and C3D)."""
import json

import numpy as np
import pytest

from gait_synthetic import walking_trial
from test_reference import write_c3d


def write_mot(path, time, names, values):
    header = f"Coordinates\nversion=1\nnRows={len(time)}\nnColumns={len(names) + 1}\ninDegrees=yes\nendheader\n"
    body = "\n".join("\t".join(f"{v:.6f}" for v in [t, *row]) for t, row in zip(time, values))
    path.write_text(header + "time\t" + "\t".join(names) + "\n" + body + "\n")


def reference_mot(tmp_path, trial, *, lag, rate=200.0, knee_bias=2.0):
    """Same motion as the trial, sampled faster on a clock that reads `lag` later, knee +bias."""
    ref_time = np.arange(-0.5, trial.time[-1] + 1, 1 / rate)
    source = ref_time - lag
    names = ["knee_angle_l", "knee_angle_r", "hip_flexion_l", "pelvis_ty"]
    values = np.column_stack([60 * np.mod(source, 1.0) + knee_bias,
                              60 * np.mod(source - 0.5, 1.0) + knee_bias,
                              20 * np.cos(2 * np.pi * source),
                              0.9 + 0.02 * np.sin(2 * np.pi * 2 * source)])
    path = tmp_path / "reference.mot"
    write_mot(path, ref_time, names, values)
    return path


def test_known_lag_gives_bias_and_offset_removed_error(cli, tmp_path):
    trial = walking_trial()
    pred = tmp_path / "trial.json"
    trial.save_json(pred)
    ref = reference_mot(tmp_path, trial, lag=0.2)
    out = tmp_path / "agreement.json"
    result = cli("benchmarks.compare", "--prediction", pred, "--reference", ref, "--mapping", "none",
                 "--lag-source", "sync event", "--lag", "0.2", "--output", out)
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    hip = data["coordinates"]["hip_flexion_l"]
    assert hip["rmse"] == pytest.approx(0, abs=0.01) and hip["coverage"] == 1.0
    # The knee ramp jumps at each heel strike, so interpolation error concentrates there; use the median.
    left = data["parameters"]["knee_flexion_peak_swing"]["left"]
    assert left["bias"] == pytest.approx(-2.0, abs=0.7)
    assert data["comparison"].startswith("agreement")


def test_estimated_lag_on_scored_angles_is_labelled_optimistic(cli, tmp_path):
    trial = walking_trial()
    pred = tmp_path / "trial.json"
    trial.save_json(pred)
    ref = reference_mot(tmp_path, trial, lag=0.137)
    out = tmp_path / "agreement.json"
    refused = cli("benchmarks.compare", "--prediction", pred, "--reference", ref, "--mapping", "none",
                  "--lag-source", "independent signals", "--estimate-lag-on", "hip_flexion_l", "--output", out)
    assert refused.returncode != 0 and "scored signals" in refused.stderr
    result = cli("benchmarks.compare", "--prediction", pred, "--reference", ref, "--mapping", "none",
                 "--lag-source", "scored signals", "--estimate-lag-on", "hip_flexion_l",
                 "--max-lag", "0.4", "--output", out, "--no-parameters")
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["lag_s"] == pytest.approx(0.137, abs=0.003)
    assert "optimistic" in data["warning"]


def test_c3d_reference_with_plugin_gait_names_and_events(cli, tmp_path):
    pytest.importorskip("c3d")
    trial = walking_trial()
    pred = tmp_path / "trial.json"
    trial.save_json(pred)
    rate, lag = 100.0, 0.3
    n = int((trial.time[-1] + 1) * rate)
    time = np.arange(n) / rate                     # frame 1 is t = 0 in the lab clock
    source = time - lag
    frames = np.zeros((n, 2, 3))
    frames[:, 0, 0] = 60 * np.mod(source, 1.0)     # LKneeAngles.X
    frames[:, 1, 0] = 20 * np.cos(2 * np.pi * source)  # LHipAngles.X
    strikes = [("Left", "Foot Strike", t + lag) for t in (1.0, 2.0, 3.0, 4.0, 5.0)]
    offs = [("Left", "Foot Off", t + lag) for t in (0.6, 1.6, 2.6, 3.6, 4.6)]
    path = write_c3d(tmp_path / "lab.c3d", labels=["Child:LKneeAngles", "Child:LHipAngles"], frames=frames,
                     rate=rate, events=strikes + offs)
    out = tmp_path / "agreement.json"
    result = cli("benchmarks.compare", "--prediction", pred, "--reference", path,
                 "--reference-name", "Plug-in Gait (generated)", "--lag-source", "sync event", "--lag", str(lag),
                 "--output", out)
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert set(data["coordinates"]) == {"knee_angle_l", "hip_flexion_l"}
    assert "LPelvisAngles.X" in data["mapping"]["disabled"]
    strike = data["events"]["left_foot_strike"]
    assert strike["matched"] == 5 and strike["missed"] == 0
    assert strike["mean_absolute_error_s"] < 0.011
