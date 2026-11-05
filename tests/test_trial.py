"""Only generated arrays and text fixtures; no real movement recordings."""
import json

import numpy as np
import pytest

from medpose.trial import Trial, load_opensim_mot


def metadata():
    return dict(recording_mode="1_phone", engine_name="synthetic", engine_version="1",
                camera_count=1, frame_rate=2)


def trial(**changes):
    args = dict(time=[0, .5], coordinate_names=("knee_angle_l", "pelvis_tx"),
                values=[[0, 1], [90, np.nan]], supports=("walker",), **metadata())
    args.update(changes)
    return Trial(**args)


def test_roundtrip(tmp_path):
    original = trial(confidence=[[1, .5], [0, np.nan]])
    path = tmp_path / "synthetic.json"
    original.save_json(path)
    assert "NaN" not in path.read_text()
    loaded = Trial.load_json(path)
    np.testing.assert_allclose(loaded.values, [[0, 1], [90, np.nan]], equal_nan=True)
    np.testing.assert_allclose(loaded.confidence, original.confidence, equal_nan=True)
    assert loaded.units == ("deg", "m")
    assert loaded.supports == ("walker",)
    assert loaded.to_dict() == original.to_dict()
    assert not loaded.values.flags.writeable


def test_empty_roundtrip():
    empty = trial(time=[], values=np.empty((0, 2)), confidence=np.empty((0, 2)))
    assert Trial.from_dict(empty.to_dict()).values.shape == (0, 2)
    assert trial().confidence is None


@pytest.mark.parametrize("changes", [
    {"time": [0, 0]}, {"time": [1, 0]}, {"time": [0, np.nan]},
    {"values": [[1]]}, {"values": [[np.inf, 0], [0, 0]]},
    {"coordinate_names": ("a", "a")}, {"coordinate_names": ()},
    {"camera_count": 2}, {"camera_count": True}, {"recording_mode": "unknown"},
    {"frame_rate": 0}, {"frame_rate": np.nan}, {"frame_rate": True},
    {"confidence": [[1.1, 0], [0, 0]]}, {"confidence": [[0]]},
    {"engine_name": ""}, {"supports": "walker"},
])
def test_boundaries(changes):
    with pytest.raises(ValueError):
        trial(**changes)


@pytest.mark.parametrize("key,value", [("schema_version", 99), ("schema_version", True),
    ("name", "forbidden"), ("units", ["rad", "m"]), ("values", [[True, 0], [0, 0]]),
    ("values", [["1", 0], [0, 0]]), ("time", [None, 1])])
def test_json_boundaries(key, value):
    data = trial().to_dict()
    data[key] = value
    with pytest.raises(ValueError):
        Trial.from_dict(data)


@pytest.mark.parametrize("text", ["", "[]", '{"schema_version":1,"schema_version":1}', '{bad'])
def test_malformed_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        Trial.load_json(path)


def test_mot_conversion(tmp_path):
    path = tmp_path / "synthetic.mot"
    path.write_text("synthetic\nnRows=2\nnColumns=3\ninDegrees=no\nendheader\n"
                    "time knee_angle_l pelvis_tx\n0 0 1\n0.5 1.5707963267948966 nan\n")
    loaded = load_opensim_mot(path, translation_unit="m", **metadata())
    # pi/2 radians = 90 degrees; translation 1 metre remains 1 metre.
    np.testing.assert_allclose(loaded.values, [[0, 1], [90, np.nan]], equal_nan=True)
    with pytest.raises(ValueError, match="Confirm pelvis"):
        load_opensim_mot(path, **metadata())
    with pytest.raises(ValueError, match="Conflicting"):
        load_opensim_mot(path, angle_unit="deg", translation_unit="m", **metadata())


@pytest.mark.parametrize("body", [
    "", "endheader\n", "endheader\nx knee\n0 1", "endheader\ntime knee\n",
    "inDegrees=yes\nendheader\ntime knee\n0 1\n1 2 3",
    "inDegrees=yes\nendheader\ntime knee\n0 bad",
    "nRows=3\ninDegrees=yes\nendheader\ntime knee\n0 1",
    "inDegrees=yes\nendheader\ntime knee\n0 1\n0 2",
    "inDegrees=unknown\nendheader\ntime knee\n0 1",
    "inDegrees=yes\ninDegrees=yes\nendheader\ntime knee\n0 1",
])
def test_malformed_mot(tmp_path, body):
    path = tmp_path / "synthetic.mot"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_opensim_mot(path, **metadata())


def test_missing_unit_and_single_frame(tmp_path):
    path = tmp_path / "synthetic.mot"
    path.write_text("endheader\ntime knee\n0 30")
    with pytest.raises(ValueError, match="units absent"):
        load_opensim_mot(path, **metadata())
    assert load_opensim_mot(path, angle_unit="deg", **metadata()).values[0, 0] == 30
