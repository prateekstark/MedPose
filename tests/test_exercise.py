import numpy as np
import pytest

from medpose.trial import Trial
from medpose.clinical.exercise import RepetitionConfig, review_exercise, side_difference, compare_exercise

CFG = RepetitionConfig(0, 8, .5, 1.1)  # simulated segmentation settings


def trial(values=(0, 5, 10, 5, 0), supports=()):
    values = np.array(values, dtype=float)
    return Trial(np.arange(len(values), dtype=float), ("knee_l", "knee_r"),
                 np.column_stack([values, values * 2]), "1_phone", "synthetic", "1", 1, 1, supports)


def review(value):
    return review_exercise(value, "knee_l", CFG, plane="sagittal", task="synthetic exercise")


def test_hand_computed_repetition():
    result = review(trial())
    # 0->10->0: range10, total4 seconds, peak t2, 2 seconds each direction.
    assert result["count"] == 1
    assert result["repetitions"][0] == dict(start=0, stop=4, peak=2, range_deg=10, duration_s=4, rising_s=2, returning_s=2)
    delta = side_difference(trial(), "knee_l", "knee_r", CFG, plane="sagittal", task="synthetic exercise")
    assert delta["left_minus_right_mean_range_deg"] == -10


@pytest.mark.parametrize("values", [(), (5, 10, 5, 0), (0, 5, 10), (0, np.nan, 10, 0), (0, 5, 7, 0)])
def test_no_invented_repetitions(values):
    assert review(trial(values))["count"] == 0


def test_context_and_invalid_threshold():
    result = compare_exercise(trial(), trial(supports=("brace",)), "knee_l", CFG,
                              plane="sagittal", task="synthetic exercise")
    assert result["label"] == "not comparable" and result["warnings"]
    with pytest.raises(ValueError):
        RepetitionConfig(8, 0, .5, 1)
