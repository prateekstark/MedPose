"""OpenCap .mot + .trc -> Trial, on generated files written in OpenCap's format."""
import numpy as np
import pytest

from medpose.engine.opencap_trial import opencap_trial, planes_for, read_trc

MARKERS = ["r_calc_study", "r_toe_study", "L_calc_study", "L_toe_study", "RASIS_study"]


def write_trc(path, time, positions, names=MARKERS, units="m"):
    """Mirror opencap-core utils.numpy2TRC (tab layout, 1-indexed frames)."""
    rate = 1 / (time[1] - time[0])
    lines = ["PathFileType  4\t(X/Y/Z) somewhere",
             "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
             f"{rate:.1f}\t{rate:.1f}\t{len(time)}\t{len(names)}\t{units}\t{rate:.1f}\t1\t{len(time)}",
             "Frame#\tTime\t" + "".join(f"{n}\t\t\t" for n in names),
             "\t\t" + "".join(f"X{i}\tY{i}\tZ{i}\t" for i in range(1, len(names) + 1)), ""]
    for f, t in enumerate(time):
        lines.append(f"{f + 1}\t{t:.8f}\t" + "".join(f"{x:.5f}\t{y:.5f}\t{z:.5f}\t" for x, y, z in positions[f]))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_mot(path, time, names, values):
    header = f"Coordinates\nversion=1\nnRows={len(time)}\nnColumns={len(names) + 1}\ninDegrees=yes\nendheader\n"
    body = "\n".join("\t".join(f"{v:.6f}" for v in [t, *row]) for t, row in zip(time, values))
    path.write_text(header + "time\t" + "\t".join(names) + "\n" + body + "\n")
    return path


@pytest.fixture
def opencap_files(tmp_path):
    time = np.arange(50) / 50
    positions = np.zeros((50, len(MARKERS), 3))
    positions[:, :, 0] = time[:, None]
    heights = {0: 0.06, 1: 0.03, 2: 0.06, 3: 0.03, 4: 0.9}  # markers above the floor
    for k, h in heights.items():
        positions[:, k, 1] = h + 0.05 * (1 + np.sin(2 * np.pi * time))[:]
    names = ["pelvis_tx", "pelvis_ty", "pelvis_tz", "knee_angle_r", "knee_angle_r_beta"]
    values = np.column_stack([time, np.full(50, 0.9), np.zeros(50), 30 * time, 30 * time])
    trc = write_trc(tmp_path / "trial.trc", time, positions)
    mot = write_mot(tmp_path / "trial.mot", time, names, values)
    return mot, trc, positions


def test_read_trc_scales_millimetres(tmp_path, opencap_files):
    _, _, positions = opencap_files
    time = np.arange(50) / 50
    path = write_trc(tmp_path / "mm.trc", time, positions * 1000, units="mm")
    t, rate, names, read = read_trc(path)
    assert rate == 50 and names == MARKERS
    np.testing.assert_allclose(read, positions, atol=1e-6)


def test_trial_has_angles_and_foot_points_on_the_floor(opencap_files):
    mot, trc, positions = opencap_files
    trial = opencap_trial(mot, trc, recording_mode="2_phone", camera_count=2, supports=("posterior walker",),
                          engine_version="test", floor_percentile=0.0)
    assert trial.engine_name == "OpenCap Core" and trial.frame_rate == 50
    assert trial.coordinate(("knee_angle_r"))[-1] == pytest.approx(30 * 49 / 50)
    feet = trial.foot_positions
    # Each marker's lowest height becomes the floor; horizontal positions are untouched.
    assert np.nanmin(feet.right[:, 0, 1]) == pytest.approx(0, abs=1e-9)
    assert np.nanmin(feet.left[:, 1, 1]) == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(feet.right[:, 0, 0], positions[:, 0, 0])
    assert "percentile" in feet.origin


def test_mismatched_files_are_refused(tmp_path, opencap_files):
    mot, _, positions = opencap_files
    other = write_trc(tmp_path / "other.trc", np.arange(50) / 40, positions)
    with pytest.raises(ValueError, match="timestamps"):
        opencap_trial(mot, other, recording_mode="2_phone", camera_count=2, engine_version="test")


def test_planes_cover_opencap_coordinates_without_guessing():
    planes = planes_for(["knee_angle_r", "knee_angle_r_beta", "pelvis_tx", "mystery"])
    assert planes == {"knee_angle_r": "sagittal", "knee_angle_r_beta": "coupled patella",
                      "pelvis_tx": "translation", "mystery": "unspecified"}


def test_cli_writes_trial_and_refuses_overwrite(cli, tmp_path, opencap_files):
    mot, trc, _ = opencap_files
    out = tmp_path / "trial.json"
    args = ["--mot", mot, "--trc", trc, "--mode", "2_phone", "--cameras", "2", "--support", "walker", "--output", out]
    first = cli("medpose.engine.opencap_trial", *args)
    assert first.returncode == 0, first.stderr
    assert "with foot points" in first.stdout and out.exists()
    second = cli("medpose.engine.opencap_trial", *args)
    assert second.returncode != 0 and "Refusing" in second.stderr
