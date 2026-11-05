"""Generated waveforms only. All chosen thresholds are synthetic test settings."""
import numpy as np
import pytest

from medpose.trial import Trial
from medpose.feet import FootPositions
from medpose.clinical.gait import ContactConfig, detect_contacts, analyse_gait


def config(scale=1):
    return ContactConfig(.05, .001, .5, .11 * scale, .2 * scale, .2 * scale)


def synthetic(scale=1, frames=61, missing_right=False, confidence=None):
    time = np.arange(frames) / 10 * scale
    phase = np.arange(frames) / 10
    # h=.05+.04*cos(pi*t): near-ground interval [0.5,1.5], then [2.5,3.5].
    # At grid boundaries force exact equality instead of floating-point cosine noise.
    height = np.round(.05 + .04 * np.cos(np.pi * phase), 10)
    left = np.zeros((frames, 2, 3)); right = left.copy()
    left[:, :, 1] = height[:, None]
    right[:, :, 1] = (.1 - height)[:, None]
    left[:, :, 0] = (time + .3)[:, None]
    right[:, :, 0] = (time - .3)[:, None]
    angles = np.column_stack([10 * np.sin(np.pi * phase), 5 * np.sin(np.pi * phase),
                              time, np.zeros(frames), np.zeros(frames)])
    feet = FootPositions(left, None if missing_right else right, [0, 1, 0], 0.,
                         "simulated sole points", confidence)
    return Trial(time, ("knee_l", "knee_r", "pelvis_tx", "pelvis_ty", "pelvis_tz"),
                 angles, "1_phone", "synthetic", "1", 1, 10 / scale, foot_positions=feet)


def analyse(trial, **kwargs):
    return analyse_gait(trial, config(), planes={n: "synthetic plane" for n in trial.coordinate_names}, **kwargs)


def test_known_events_metrics_and_asymmetry():
    trial = synthetic()
    events = detect_contacts(trial, "left", config())
    assert events["initial_contacts"] == [5, 25, 45]
    assert events["departures"] == [16, 36, 56]  # first sample strictly above threshold
    assert len(events["cycles"]) == 2
    result = analyse(trial, progression_direction=[1, 0, 0])
    left = result["sides"]["left"]
    # Period 2 s -> 120/2 = 60 steps/min; 1.1 s proximity / 2 = 55%.
    assert left["cadence_steps_min"]["mean"] == 60
    assert left["stance_percent"]["mean"] == pytest.approx(55)
    assert left["coordinates"]["knee_l"]["range"]["mean"] == pytest.approx(20)
    assert len(left["coordinates"]["knee_l"]["normalized"][0]) == 101
    assert result["differences"]["knee"]["left_minus_right_range"] == pytest.approx(10)
    assert left["duration_s"]["sd"] == 0
    # Pelvis x=t metres -> speed 1 m/s. Heel separation = .3-(-.3)=.6 m.
    assert result["speed"]["sides"]["left"]["mean"] == pytest.approx(1)
    assert result["step_length"]["sides"]["left"]["mean"] == pytest.approx(.6)
    assert result["speed"]["error_label"] == "error not yet measured"


def test_slow_gait_has_no_frequency_cutoff():
    result = detect_contacts(synthetic(scale=10), "left", config(scale=10))
    assert result["initial_contacts"] == [5, 25, 45]
    assert len(result["cycles"]) == 2


def test_missing_side_short_clip_and_empty():
    trial = synthetic(missing_right=True)
    assert detect_contacts(trial, "right", config())["status"] == "detection failed"
    assert analyse(trial)["differences"]["knee"]["left_minus_right_range"] is None
    assert not detect_contacts(synthetic(frames=15), "left", config())["cycles"]
    assert not detect_contacts(synthetic(frames=0), "left", config())["cycles"]


def test_low_confidence_does_not_bridge():
    confidence = np.ones((61, 2)); confidence[10] = .1
    trial = synthetic(confidence=confidence)
    events = detect_contacts(trial, "left", config())
    assert events["low_confidence_frames"] == [10]
    assert len(events["cycles"]) == 1  # first cycle crosses unreliable frame and is rejected
    assert events["cycles"][0]["start"] == 25


def test_nan_positions_reject_cycle():
    trial = synthetic()
    positions = trial.foot_positions.left.copy(); positions[10] = np.nan
    feet = FootPositions(positions, None, [0, 1, 0], 0., "simulated")
    data = trial.to_dict(); data["foot_positions"] = feet.to_dict()
    events = detect_contacts(Trial.from_dict(data), "left", config())
    assert events["invalid_frames"] == [10]
    assert len(events["cycles"]) == 1


def test_missing_angle_not_interpolated_and_variability():
    data = synthetic().to_dict()
    data["values"][10][0] = None
    result = analyse(Trial.from_dict(data))
    metric = result["sides"]["left"]["coordinates"]["knee_l"]
    assert metric["normalized"][0] is None
    assert metric["range"]["count"] == 1
    assert metric["range"]["sd"] is None
    data = synthetic().to_dict()
    # Second complete left cycle amplitude doubled: ranges 20 and 40, mean30,
    # sample SD sqrt(((20-30)^2+(40-30)^2)/(2-1)) = sqrt(200).
    for index, row in enumerate(data["values"]):
        row[0] = float(10 * np.cos(np.pi * index / 10))
    for row in data["values"][25:46]:
        row[0] *= 2
    metric = analyse(Trial.from_dict(data))["sides"]["left"]["coordinates"]["knee_l"]
    assert metric["range"]["mean"] == pytest.approx(30)
    assert metric["range"]["sd"] == pytest.approx(np.sqrt(200))


def test_schema_v1_v2_and_foot_roundtrip(tmp_path):
    trial = synthetic()
    path = tmp_path / "simulated.json"
    trial.save_json(path)
    loaded = Trial.load_json(path)
    np.testing.assert_allclose(loaded.foot_positions.left, trial.foot_positions.left)
    data = trial.to_dict(); data.pop("foot_positions"); data["schema_version"] = 1
    assert Trial.from_dict(data).foot_positions is None
    data["schema_version"] = 99
    with pytest.raises(ValueError):
        Trial.from_dict(data)


@pytest.mark.parametrize("normal", [[0, 2, 0], [0, np.nan, 0], [0, 0]])
def test_bad_ground_plane(normal):
    with pytest.raises(ValueError):
        FootPositions(None, None, normal, 0, "simulated")


def test_bad_frames_and_geometry():
    data = synthetic().to_dict()
    data["foot_positions"]["left"] = []
    with pytest.raises(ValueError):
        Trial.from_dict(data)
    with pytest.raises(ValueError):
        analyse(synthetic(), progression_direction=[0, 1, 0])
    with pytest.raises(ValueError):
        ContactConfig(-1, 0, .5, .1, .2, .2)
