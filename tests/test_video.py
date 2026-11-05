import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, ffmpeg
from medpose.capture.video import (
    check_intrinsics_match,
    intrinsics_size,
    load_intrinsics,
    nominal_fps,
    normalize,
    probe,
)


def test_probe_reads_size_and_rate(make_video):
    info = probe(make_video(size=(320, 240), fps=30))
    assert (info["width"], info["height"]) == (320, 240)
    assert info["fps"] == pytest.approx(30, abs=0.5)
    assert info["variable_frame_rate"] is False


def test_rotated_android_video_becomes_upright(make_video, tmp_path):
    source = make_video(size=(320, 240))
    rotated = tmp_path / "rotated.mp4"
    ffmpeg("-display_rotation", "90", "-i", source, "-c", "copy", rotated)
    before = probe(rotated)
    assert before["rotation"] != 0
    assert (before["width"], before["height"]) == (240, 320)

    after = normalize(rotated, tmp_path / "upright.mov")
    assert after["rotation"] == 0
    assert (after["width"], after["height"]) == (240, 320)


def test_variable_frame_rate_is_detected_and_fixed(make_video, tmp_path):
    vfr = make_video("vfr.mp4", seconds=2, extra=("-vf", "select='mod(n\\,5)'", "-fps_mode", "vfr"))
    assert probe(vfr)["variable_frame_rate"] is True

    fixed = normalize(vfr, tmp_path / "fixed.mov", fps=30)
    assert fixed["variable_frame_rate"] is False
    assert fixed["fps"] == pytest.approx(30, abs=0.5)


def test_webcam_style_mkv_is_accepted(make_video, tmp_path):
    source = make_video("webcam.mkv", extra=("-c:v", "mjpeg", "-q:v", "5"))
    assert normalize(source, tmp_path / "webcam.mov")["rotation"] == 0


def test_zero_byte_video_is_refused(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="could not read"):
        probe(empty)
    with pytest.raises(RuntimeError, match="could not read"):
        normalize(empty, tmp_path / "out.mov")
    assert not (tmp_path / "out.mov").exists()


def test_missing_or_none_video_is_refused(tmp_path):
    with pytest.raises(RuntimeError):
        probe(tmp_path / "missing.mp4")
    with pytest.raises(RuntimeError):
        probe(None)


def test_single_frame_video_has_no_timing_to_misread(make_video):
    info = probe(make_video(extra=("-frames:v", "1")))
    assert info["frames"] == 1
    assert info["variable_frame_rate"] is False


def test_very_small_video_is_read(make_video):
    info = probe(make_video(size=(2, 2)))
    assert (info["width"], info["height"]) == (2, 2)


@pytest.mark.parametrize("measured, expected", [(29.97, 30), (59.4, 59), (0.0, 1), (0.4, 1)])
def test_nominal_fps_snaps_to_a_usable_whole_number(measured, expected):
    assert nominal_fps(measured) == expected


def test_zero_fps_means_keep_the_source_rate(make_video, tmp_path):
    assert round(normalize(make_video(fps=30), tmp_path / "out.mov", fps=0)["fps"]) == 30


def test_negative_fps_is_refused_and_leaves_no_file(make_video, tmp_path):
    target = tmp_path / "out.mov"
    with pytest.raises(RuntimeError, match="could not normalize"):
        normalize(make_video(), target, fps=-1)
    assert not target.exists()


def test_empty_missing_or_none_intrinsics_are_refused(tmp_path):
    empty = tmp_path / "empty.pickle"
    empty.write_bytes(b"")
    for path in (empty, tmp_path / "missing.pickle", None):
        with pytest.raises(ValueError, match="Could not read"):
            intrinsics_size(path)


def test_intrinsics_without_an_image_size_are_refused(tmp_path):
    path = tmp_path / "no_size.pickle"
    path.write_bytes(pickle.dumps({"intrinsicMat": np.eye(3)}))
    with pytest.raises(ValueError, match="Could not read"):
        intrinsics_size(path)


def test_normalize_requires_mov(make_video, tmp_path):
    with pytest.raises(ValueError, match=".mov"):
        normalize(make_video(), tmp_path / "out.mp4")


def test_size_check_explains_orientation_mismatch(tmp_path, write_intrinsics):
    profile = write_intrinsics(tmp_path / "cam.pickle", 1080, 1920)
    assert intrinsics_size(profile) == (1080, 1920)
    check_intrinsics_match(profile, {"width": 1080, "height": 1920})
    with pytest.raises(ValueError, match="different orientation"):
        check_intrinsics_match(profile, {"width": 1920, "height": 1080})
    with pytest.raises(ValueError, match="same lens and resolution"):
        check_intrinsics_match(profile, {"width": 720, "height": 1280})


class _Malicious:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


def test_tampered_intrinsics_cannot_run_code(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "bad.pickle"
    path.write_bytes(pickle.dumps(_Malicious(marker)))
    with pytest.raises(ValueError, match="Could not read"):
        intrinsics_size(path)
    assert not marker.exists()


def test_bundled_iphone_profile_still_loads():
    profiles = sorted((ROOT / "opencap-core" / "CameraIntrinsics").glob("*/Deployed*/cameraIntrinsics.pickle"))
    if not profiles:
        pytest.skip("OpenCap Core checkout not present")
    assert len(np.asarray(load_intrinsics(profiles[0])["imageSize"]).flatten()) == 2
