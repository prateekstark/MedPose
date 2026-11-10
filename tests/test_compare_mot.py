import importlib.util

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("compare_mot", ROOT / "tools" / "compare_mot.py")
compare_mot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_mot)


def write_mot(path, rows, columns=("time", "knee_angle_r", "pelvis_ty")):
    body = "\n".join("\t".join(str(v) for v in row) for row in rows)
    path.write_text("Coordinates\nversion=1\nendheader\n" + "\t".join(columns) + "\n" + body + "\n")
    return path


def test_identical_files_have_zero_difference(tmp_path):
    a = write_mot(tmp_path / "a.mot", [(0, 10, 1), (0.1, 20, 1)])
    assert compare_mot.compare(a, a)["rms"] == {"knee_angle_r": 0.0, "pelvis_ty": 0.0}


def test_rms_matches_a_hand_computed_value(tmp_path):
    a = write_mot(tmp_path / "a.mot", [(0, 10, 1), (0.1, 20, 1)])
    b = write_mot(tmp_path / "b.mot", [(0, 13, 1), (0.1, 16, 1)])  # differences 3 and 4
    assert compare_mot.compare(a, b)["rms"]["knee_angle_r"] == pytest.approx((12.5) ** 0.5)


def test_uneven_lengths_compare_shared_frames_and_report_both(tmp_path):
    a = write_mot(tmp_path / "a.mot", [(0, 10, 1), (0.1, 20, 1), (0.2, 99, 1)])
    b = write_mot(tmp_path / "b.mot", [(0, 10, 1), (0.1, 20, 1)])
    result = compare_mot.compare(a, b)
    assert (result["frames_a"], result["frames_b"]) == (3, 2)
    assert result["rms"]["knee_angle_r"] == 0.0


def test_single_frame_file_is_read(tmp_path):
    a = write_mot(tmp_path / "a.mot", [(0, 10, 1)])
    assert compare_mot.compare(a, a)["frames_a"] == 1


def test_different_coordinates_are_refused(tmp_path):
    a = write_mot(tmp_path / "a.mot", [(0, 10, 1)])
    b = write_mot(tmp_path / "b.mot", [(0, 10, 1)], columns=("time", "hip_flexion_r", "pelvis_ty"))
    with pytest.raises(ValueError, match="different coordinates"):
        compare_mot.compare(a, b)


def test_file_with_a_header_but_no_rows_is_refused(tmp_path):
    path = write_mot(tmp_path / "empty.mot", [])
    with pytest.raises(ValueError, match="no data rows"):
        compare_mot.read_mot(path)


def test_non_numeric_row_is_refused(tmp_path):
    path = write_mot(tmp_path / "bad.mot", [(0, "oops", 1)])
    with pytest.raises(ValueError):
        compare_mot.read_mot(path)


def test_file_without_a_header_is_refused(tmp_path):
    path = tmp_path / "bad.mot"
    path.write_text("")
    with pytest.raises(ValueError, match="endheader"):
        compare_mot.read_mot(path)


def test_different_frame_rates_are_matched_by_time_not_row(tmp_path):
    # b samples the same ramp at twice the rate; row-by-row comparison would report large errors.
    a = write_mot(tmp_path / "a.mot", [(t / 10, 100 * t / 10, 1) for t in range(5)])
    b = write_mot(tmp_path / "b.mot", [(t / 20, 100 * t / 20, 1) for t in range(9)])
    result = compare_mot.compare(a, b)
    assert result["frames_compared"] == 5
    assert result["rms"]["knee_angle_r"] == pytest.approx(0.0, abs=1e-9)


def test_files_without_shared_time_are_refused(tmp_path):
    a = write_mot(tmp_path / "a.mot", [(0, 1, 1), (0.1, 2, 1)])
    b = write_mot(tmp_path / "b.mot", [(5, 1, 1), (5.1, 2, 1)])
    with pytest.raises(ValueError, match="share no time range"):
        compare_mot.compare(a, b)
