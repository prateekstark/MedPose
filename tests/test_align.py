"""Time alignment on generated signals with known offsets."""
import numpy as np
import pytest

from benchmarks.align import align_reference, estimate_lag, resample
from medpose.trial import Trial


def test_resample_keeps_exact_samples_and_interpolates_between():
    out = resample([0, 1, 2], [[0.0], [10.0], [20.0]], [0, 0.5, 2], max_gap_s=1.5)
    assert out[:, 0].tolist() == [0.0, 5.0, 20.0]


def test_resample_never_extrapolates_or_bridges_gaps_or_missing_values():
    out = resample([0, 1, 5, 6], [0.0, 1.0, 5.0, np.nan], [-1, 0.5, 3, 5, 5.5, 7], max_gap_s=1.5)[:, 0]
    assert np.isnan(out[0]) and np.isnan(out[5])      # outside range
    assert out[1] == 0.5
    assert np.isnan(out[2])                            # 1 -> 5 is a 4 s gap
    assert out[3] == 5.0                               # exact sample survives a NaN neighbour
    assert np.isnan(out[4])                            # interpolating into a NaN stays NaN


def test_resample_first_sample_with_missing_neighbour_is_kept():
    assert resample([0, 1], [3.0, np.nan], [0.0], max_gap_s=2)[0, 0] == 3.0


def signal(t):
    return np.sin(2 * np.pi * 0.9 * t) + 0.4 * np.sin(2 * np.pi * 2.3 * t + 1)


def test_estimate_lag_recovers_known_clock_offset_across_rates():
    lag = 0.137
    pred_time = np.arange(0, 6, 1 / 60)
    ref_time = np.arange(-1, 8, 1 / 200)
    # The reference clock reads `lag` later for the same instant.
    result = estimate_lag(pred_time, signal(pred_time), ref_time, signal(ref_time - lag),
                          max_lag_s=0.5, step_s=0.001, max_gap_s=0.02)
    assert result["lag_s"] == pytest.approx(lag, abs=0.0015)
    assert result["correlation"] > 0.999
    assert result["warnings"] == []


def test_lag_outside_the_window_is_flagged():
    pred_time = np.arange(0, 6, 1 / 60)
    ref_time = np.arange(-2, 9, 1 / 200)
    result = estimate_lag(pred_time, signal(pred_time), ref_time, signal(ref_time - 0.8),
                          max_lag_s=0.3, step_s=0.01, max_gap_s=0.02)
    assert any("search edge" in w for w in result["warnings"])


def test_lag_search_refuses_mismatched_channels():
    with pytest.raises(ValueError, match="same channels"):
        estimate_lag([0, 1, 2], np.zeros((3, 2)), [0, 1, 2], np.zeros((3, 1)), max_lag_s=1, step_s=0.1, max_gap_s=1)


def trial(time, names, values, engine="engine"):
    return Trial(time, names, values, "1_phone", engine, "1", 1, 60.0)


def test_align_reference_puts_reference_on_prediction_clock():
    pred_time = np.arange(0, 2, 1 / 60)
    ref_time = np.arange(0, 3, 1 / 200)
    lag = 0.25
    pred = trial(pred_time, ("knee_angle_r", "hip_flexion_r"), np.column_stack([signal(pred_time), pred_time]))
    ref = trial(ref_time, ("knee_angle_r", "ankle_angle_r"), np.column_stack([signal(ref_time - lag), ref_time]),
                engine="lab")
    p, r = align_reference(pred, ref, lag_s=lag, max_gap_s=0.01)
    assert p.coordinate_names == r.coordinate_names == ("knee_angle_r",)
    assert np.array_equal(p.time, r.time)
    assert r.engine_name == "lab"
    np.testing.assert_allclose(r.values[:, 0], p.values[:, 0], atol=2e-3)


def test_align_reference_requires_shared_coordinates():
    a = trial([0, 1], ("knee_angle_r",), [[0], [1]])
    b = trial([0, 1], ("hip_flexion_r",), [[0], [1]])
    with pytest.raises(ValueError, match="share no coordinates"):
        align_reference(a, b, lag_s=0, max_gap_s=1)
