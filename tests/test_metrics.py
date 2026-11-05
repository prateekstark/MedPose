"""Hand-computed metric checks; no data downloads or optional ML packages."""
import numpy as np
import pytest

from benchmarks.metrics import (
    bias, bias_corrected_mae, mae, median_absolute_error, rmse, subject_bootstrap_ci, wrapped_difference,
)


def test_median_differs_from_mean_when_one_frame_is_an_outlier():
    # Absolute errors 1, 2, 10: median=2; mean=13/3.
    pred, true = [[1], [-2], [10]], [[0], [0], [0]]
    assert median_absolute_error(pred, true).values[0] == 2
    assert mae(pred, true).values[0] == pytest.approx(13 / 3)


def test_median_excludes_missing_frames_and_reports_them():
    # Column 0 errors 1, 3: median=2. Column 1 errors 2, 4: median=3. Column 2 all missing.
    result = median_absolute_error([[1, -2, np.nan], [3, 4, np.nan], [np.nan, 100, np.nan]],
                                   [[0, 0, 0], [0, 0, np.nan], [0, np.nan, 0]])
    np.testing.assert_allclose(result.values, [2, 3, np.nan], equal_nan=True)
    np.testing.assert_array_equal(result.excluded_frames, [1, 1, 3])


def test_median_of_an_empty_trial_is_nan():
    result = median_absolute_error(np.empty((0, 2)), np.empty((0, 2)))
    assert np.isnan(result.values).all()
    np.testing.assert_array_equal(result.excluded_frames, [0, 0])


@pytest.mark.parametrize("function, expected", [
    (mae, [2, 3, np.nan]),             # |[1,3]| mean=2; |[-2,4]| mean=3
    (rmse, [np.sqrt(5), np.sqrt(10), np.nan]),  # (1+9)/2; (4+16)/2
    (bias, [2, 1, np.nan]),            # (1+3)/2=2; (-2+4)/2=1
    (bias_corrected_mae, [1, 3, np.nan]),  # centered [-1,1], [-3,3]
])
def test_hand_computed_paired_nan_metrics(function, expected):
    pred = [[1, -2, np.nan], [3, 4, np.nan], [np.nan, 100, np.nan]]
    true = [[0, 0, 0], [0, 0, np.nan], [0, np.nan, 0]]
    result = function(pred, true)
    np.testing.assert_allclose(result.values, expected, equal_nan=True)
    np.testing.assert_array_equal(result.excluded_frames, [1, 1, 3])


def test_raw_and_corrected_are_distinct_and_trial_specific():
    # Constant offset 5: raw MAE=5, bias=5, within-trial corrected MAE=0.
    pred, true = [[6], [7]], [[1], [2]]
    assert mae(pred, true).values[0] == 5
    assert bias(pred, true).values[0] == 5
    assert bias_corrected_mae(pred, true).values[0] == 0


def test_wraparound_is_explicit():
    result = wrapped_difference([[179, -179, 180, np.nan]], [[-179, 179, 0, 1]])
    # 358 wraps to -2; -358 wraps to 2; antipodal tie is -180.
    np.testing.assert_allclose(result.values, [[-2, 2, -180, np.nan]], equal_nan=True)
    np.testing.assert_array_equal(result.excluded_frames, [0, 0, 0, 1])
    assert mae([[179]], [[-179]]).values[0] == 358


@pytest.mark.parametrize("function", [mae, rmse, bias, bias_corrected_mae, wrapped_difference])
@pytest.mark.parametrize("pred,true", [([[1]], [[1, 2]]), ([1], [1]), ([[np.inf]], [[0]])])
def test_invalid_inputs(function, pred, true):
    with pytest.raises(ValueError):
        function(pred, true)


def test_empty_trial():
    result = mae(np.empty((0, 2)), np.empty((0, 2)))
    assert np.isnan(result.values).all()
    np.testing.assert_array_equal(result.excluded_frames, [0, 0])


def test_bootstrap_subject_weights_and_reproducibility():
    # Subject means are 0 and 10, so equal-subject point=5 (pooled frames=2.5).
    # Resampling two subjects yields only means 0, 5, 10; 95% endpoints=0,10.
    data = {"a": [0, 0, 0], "b": [10]}
    first = subject_bootstrap_ci(data, n_boot=10000, seed=17)
    assert first == (5.0, 0.0, 10.0)
    assert first == subject_bootstrap_ci(data, n_boot=10000, seed=17)


def test_bootstrap_matches_whole_subject_sampling():
    # Means [1, 4, 9]; point=(1+4+9)/3=14/3. Independently draw subject IDs.
    data = [[0, 2], [4], [8, 9, 10]]
    rng = np.random.default_rng(12)
    draws = rng.integers(0, 3, size=(100, 3))
    expected = np.quantile(np.array([1, 4, 9])[draws].mean(axis=1), [.1, .9])
    point, low, high = subject_bootstrap_ci(data, n_boot=100, ci=.8, seed=12)
    assert point == pytest.approx(14 / 3)
    np.testing.assert_allclose([low, high], expected)


def test_bootstrap_nan_counts_and_single_subject():
    with pytest.warns(RuntimeWarning, match=r"\[1\]"):
        # [2,NaN,4] gives subject mean 3; every subject resample gives 3.
        assert subject_bootstrap_ci([[2, np.nan, 4]], n_boot=10) == (3, 3, 3)


@pytest.mark.parametrize("data,kwargs", [
    ([], {}), ([[np.nan]], {}), ([[]], {}), ([[np.inf]], {}),
    ([1], {"n_boot": 0}), ([1], {"n_boot": 1.5}), ([1], {"n_boot": True}),
    ([1], {"ci": 1}), ([1], {"ci": 0}), ([1], {"ci": np.nan}),
])
def test_bootstrap_invalid_inputs(data, kwargs):
    with pytest.raises(ValueError):
        subject_bootstrap_ci(data, **kwargs)
