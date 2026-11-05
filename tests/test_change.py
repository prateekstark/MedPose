"""Hand-calculated synthetic repeatability; no clinical measurements."""
import numpy as np
import pytest

from medpose.trial import Trial
from medpose.clinical.change import SessionMeasure, compare_change, icc_absolute_agreement


def measure(value, supports=(), mode="1_phone"):
    trial = Trial([0], ("knee",), [[value]], mode, "synthetic", "1",
                  1 if mode == "1_phone" else 2, 30, supports)
    return SessionMeasure(trial, "synthetic knee mean", "deg", float(value))


def compare(before=0, after=4, repeats=(0, 2, 4), **changes):
    kwargs = dict(source="measured", min_repeat_sessions=3, stable_baseline_confirmed=True)
    kwargs.update(changes)
    return compare_change(measure(before), measure(after), [measure(v) for v in repeats], **kwargs)


def test_icc_hand_calculation():
    # [[1,2],[3,4],[5,6]]: MS_subject=8, MS_session=1.5, MSE=0.
    # ICC(A,1)=8/(8+(2/3)*1.5)=8/9 (consistency would misleadingly be 1).
    result = icc_absolute_agreement([[1, 2], [3, 4], [5, 6]], source="simulated")
    assert result["value"] == pytest.approx(8 / 9)
    assert result["source"] == "simulated"
    assert icc_absolute_agreement([[1, 1], [1, 1]], source="simulated")["value"] is None


def test_personal_mdc_and_strict_boundary():
    # Baseline [0,2,4]: mean2, sample SD=sqrt(8/2)=2.
    # MDC95=1.96*sqrt(2)*2. Strictly equal does not exceed.
    mdc = 1.96 * np.sqrt(2) * 2
    result = compare(after=float(mdc))
    assert result["mdc95"] == pytest.approx(mdc)
    assert result["label"] == "does not exceed measured noise"
    assert compare(after=float(mdc + .1))["label"] == "exceeds measured noise"
    assert compare(after=10, source="simulated")["label"] == "noise not measured"


def test_missing_sessions_and_degenerate_noise():
    assert compare(repeats=())["label"] == "not enough sessions"
    result = compare(repeats=(0, 2, np.nan))
    assert result["label"] == "not enough sessions" and result["excluded_sessions"] == 1
    assert compare(repeats=(1, 1, 1))["label"] == "noise estimate degenerate"
    assert compare(after=np.nan)["label"] == "comparison value missing"
    assert compare(stable_baseline_confirmed=False)["label"] == "stable baseline not confirmed"


@pytest.mark.parametrize("change", [{"supports": ("walker",)}, {"mode": "2_phone"}])
def test_context_mismatch_suppresses_label(change):
    result = compare_change(measure(0), measure(100, **change), [measure(v) for v in (0, 2, 4)],
                            source="measured", min_repeat_sessions=3, stable_baseline_confirmed=True)
    assert result["label"] == "not comparable"
    assert result["warnings"] and result["mdc95"] is None


def test_icc_missing_and_malformed():
    result = icc_absolute_agreement([[1, 2], [3, 4], [np.nan, 4]], source="simulated")
    assert result["excluded_subjects"] == 1
    assert icc_absolute_agreement(np.empty((0, 2)), source="simulated")["value"] is None
    for value in ([1, 2], [[np.inf, 1]]):
        with pytest.raises(ValueError):
            icc_absolute_agreement(value, source="simulated")
    with pytest.raises(ValueError):
        compare(min_repeat_sessions=1)


def test_cannot_reuse_baseline_trial():
    same = measure(1)
    with pytest.raises(ValueError):
        compare_change(measure(0), measure(10), [same, same], source="simulated",
                       min_repeat_sessions=2, stable_baseline_confirmed=True)
