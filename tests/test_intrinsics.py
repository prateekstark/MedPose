import numpy as np
import pytest

from conftest import render_checkerboard_video
from medpose.capture.intrinsics import (
    DEFAULT_MAX_RMS_PX,
    DEFAULT_MIN_COVERAGE,
    MIN_VIEWS,
    image_coverage,
    quality_problems,
)
from medpose.capture.video import load_intrinsics


def test_coverage_counts_touched_cells():
    corner_only = [np.array([[[1.0, 1.0]], [[2.0, 2.0]]], dtype=np.float32)]
    assert image_coverage(corner_only, (600, 400)) == pytest.approx(1 / 24)
    everywhere = [np.array([[[x + 50.0, y + 50.0]] for x in range(0, 600, 100) for y in range(0, 400, 100)],
                           dtype=np.float32)]
    assert image_coverage(everywhere, (600, 400)) == pytest.approx(1.0)


def test_quality_gate_reports_each_failure():
    assert quality_problems(rms=0.4, views=20, coverage=0.8) == []
    assert len(quality_problems(rms=2.0, views=3, coverage=0.1)) == 3
    assert "reprojection" in quality_problems(rms=2.0, views=20, coverage=0.8)[0]
    assert "covered" in quality_problems(rms=0.4, views=20, coverage=0.1)[0]


def test_coverage_of_no_detections_is_zero():
    assert image_coverage([], (600, 400)) == 0.0


def test_coverage_keeps_points_on_the_far_edge_in_the_last_cell():
    on_edge = [np.array([[[600.0, 400.0]]], dtype=np.float32)]
    assert image_coverage(on_edge, (600, 400)) == pytest.approx(1 / 24)


def test_quality_gate_passes_exactly_at_each_limit():
    assert quality_problems(rms=DEFAULT_MAX_RMS_PX, views=MIN_VIEWS, coverage=DEFAULT_MIN_COVERAGE) == []


def test_quality_gate_fails_just_past_each_limit():
    assert quality_problems(rms=DEFAULT_MAX_RMS_PX + 0.001, views=MIN_VIEWS, coverage=1.0)
    assert quality_problems(rms=0.1, views=MIN_VIEWS - 1, coverage=1.0)
    assert quality_problems(rms=0.1, views=MIN_VIEWS, coverage=DEFAULT_MIN_COVERAGE - 0.001)


def test_quality_gate_reports_no_detections():
    assert len(quality_problems(rms=0.0, views=0, coverage=0.0)) == 2


def test_video_without_a_board_saves_nothing(tmp_path, make_video, cli):
    output = tmp_path / "cam.pickle"
    result = cli("medpose.capture.intrinsics", make_video(), "--corners", 9, 6, "--square-mm", 30,
                 "--output", output)
    assert result.returncode != 0
    assert "usable board views" in result.stderr
    assert not output.exists()


def test_empty_video_file_saves_nothing(tmp_path, cli):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    output = tmp_path / "cam.pickle"
    result = cli("medpose.capture.intrinsics", empty, "--corners", 9, 6, "--square-mm", 30, "--output", output)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("flags", [("--square-mm", 0), ("--square-mm", -30), ("--square-mm", 30, "--frame-step", 0)])
def test_invalid_settings_are_refused(tmp_path, make_video, cli, flags):
    result = cli("medpose.capture.intrinsics", make_video(), "--corners", 9, 6, *flags,
                 "--output", tmp_path / "cam.pickle")
    assert result.returncode != 0
    assert "positive square size" in result.stderr


def test_calibration_recovers_known_camera(tmp_path, cli):
    video = render_checkerboard_video(tmp_path / "board.mp4", focal=800.0)
    output = tmp_path / "cam.pickle"
    result = cli("medpose.capture.intrinsics", video, "--corners", 9, 6, "--square-mm", 30,
                 "--frame-step", 2, "--output", output)
    assert result.returncode == 0, result.stderr + result.stdout
    params = load_intrinsics(output)
    k = params["intrinsicMat"]
    assert k[0, 0] == pytest.approx(800, rel=0.03)
    assert k[1, 1] == pytest.approx(800, rel=0.03)
    assert k[0, 2] == pytest.approx(480, abs=15)
    assert k[1, 2] == pytest.approx(360, abs=15)
    assert params["imageSize"].flatten().tolist() == [720, 960]


def test_rejected_calibration_saves_nothing(tmp_path, cli):
    video = render_checkerboard_video(tmp_path / "board.mp4")
    output = tmp_path / "cam.pickle"
    result = cli("medpose.capture.intrinsics", video, "--corners", 9, 6, "--square-mm", 30,
                 "--frame-step", 2, "--output", output, "--max-rms", 0.00001)
    assert result.returncode == 1
    assert "nothing saved" in result.stderr
    assert not output.exists()
