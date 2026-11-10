"""Reference loaders on generated C3D/CSV files; no real lab data."""
import json
import warnings

import numpy as np
import pytest

from benchmarks.reference import (c3d_channels, load_mapping, read_c3d, read_table, reference_events,
                                  reference_trial)

c3d = pytest.importorskip("c3d")


def write_c3d(path, *, labels, frames, rate=100.0, first_frame=1, events=(), invalid=()):
    """frames: (n, labels, 3). events: (context, label, seconds). invalid: (frame, label index)."""
    writer = c3d.Writer(point_rate=rate)
    rows = []
    for i, frame in enumerate(frames):
        points = np.zeros((len(labels), 5))
        points[:, :3] = frame
        for f, j in invalid:
            if f == i:
                points[j, 3] = -1
        rows.append((points, np.zeros((0, 0))))
    writer.add_frames(rows)
    writer.set_point_labels(list(labels))
    writer.set_start_frame(first_frame)
    if events:
        group = writer.add_group(10, "EVENT", "events")
        pack = lambda xs: "".join(x.ljust(16) for x in xs)
        group.add_str("LABELS", "", pack([e[1] for e in events]), 16, len(events))
        group.add_str("CONTEXTS", "", pack([e[0] for e in events]), 16, len(events))
        times = np.array([[t // 60, t % 60] for _, _, t in events], dtype=np.float32)
        group.add_array("TIMES", "", times)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "wb") as handle:
            writer.write(handle)
    return path


def test_c3d_angles_events_prefixes_and_invalid_samples(tmp_path):
    n = 20
    frames = np.zeros((n, 2, 3))
    frames[:, 0, 0] = np.arange(n)          # LKneeAngles X
    frames[:, 1, 1] = -np.arange(n)         # RHipAngles Y
    path = write_c3d(tmp_path / "lab.c3d", labels=["Child:LKneeAngles", "RHipAngles"], frames=frames,
                     first_frame=11, events=[("Left", "Foot Strike", 0.15), ("Right", "Foot Off", 0.18)],
                     invalid=[(3, 0)])
    data = read_c3d(path)
    assert set(data["points"]) == {"LKneeAngles", "RHipAngles"}
    assert data["time"][0] == pytest.approx(0.10)  # frame 11 at 100 Hz, frame 1 is t = 0
    assert np.isnan(data["points"]["LKneeAngles"][3]).all()
    assert data["points"]["LKneeAngles"][4, 0] == 4
    channels = c3d_channels(data)
    assert channels["RHipAngles.Y"][5] == -5
    grouped = reference_events(data["events"])
    assert grouped["left"]["foot_strike"] == pytest.approx([0.15])
    assert grouped["right"]["foot_off"] == pytest.approx([0.18])


def test_mapping_uses_enabled_and_caution_and_skips_disabled():
    time = np.arange(5) / 100
    channels = {"LKneeAngles.X": np.arange(5.0), "LHipAngles.Y": np.ones(5), "LPelvisAngles.X": np.zeros(5)}
    trial, report = reference_trial(time, channels, "plugin_gait", recording_mode="2_phone", camera_count=2,
                                    frame_rate=60.0, reference_name="Plug-in Gait (synthetic)")
    assert trial.coordinate_names == ("knee_angle_l", "hip_adduction_l")
    assert report["used"] == ["knee_angle_l"] and report["caution"] == ["hip_adduction_l"]
    assert "LPelvisAngles.X" in report["disabled"]
    sagittal_only, _ = reference_trial(time, channels, "plugin_gait", recording_mode="2_phone", camera_count=2,
                                       frame_rate=60.0, reference_name="x", include_caution=False)
    assert sagittal_only.coordinate_names == ("knee_angle_l",)


def test_mapping_files_are_validated(tmp_path):
    data = load_mapping("plugin_gait")
    bad = dict(data, entries=[dict(data["entries"][0], sign=None)])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="sign"):
        load_mapping(path)


def test_shipped_mapping_never_enables_pelvis():
    entries = {e["coordinate"]: e for e in load_mapping()["entries"]}
    assert all(entries[c]["status"] == "disabled" for c in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"))


def test_read_table(tmp_path):
    path = tmp_path / "lab.csv"
    path.write_text("time,knee_angle_l\n0,1\n0.01,\n0.02,3\n")
    time, columns = read_table(path)
    assert time.tolist() == [0, 0.01, 0.02]
    assert np.isnan(columns["knee_angle_l"][1])
