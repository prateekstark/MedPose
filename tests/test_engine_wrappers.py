import pytest
import yaml

from medpose.capture.video import probe

STAGE = "medpose.engine.core_stage"
CORE_RUN = "medpose.engine.core_run"
# Plausible child measurements: synthetic values, not from any real person.
HEIGHT_M = 1.3
MASS_KG = 28.5
CAMERAS = ("Cam0", "Cam1")


def session_file(tmp_path, **overrides):
    data = {"height_m": HEIGHT_M, "mass_kg": MASS_KG, "gender_mf": "m",
            "iphoneModel": {"Cam0": "android_a", "Cam1": "android_b"}}
    data.update(overrides)
    data = {k: v for k, v in data.items() if v is not ...}
    path = tmp_path / "sessionMetadata.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def core_with_profiles(tmp_path, write_intrinsics, size=(320, 240)):
    core = tmp_path / "core"
    for model in ("android_a", "android_b"):
        write_intrinsics(core / "CameraIntrinsics" / model / "Deployed" / "cameraIntrinsics.pickle", *size)
    return core


def test_stages_two_phones_at_the_slowest_rate(tmp_path, make_video, cli, write_intrinsics):
    core = core_with_profiles(tmp_path, write_intrinsics)
    cam0, cam1 = make_video("a.mp4", fps=30), make_video("b.mp4", fps=60)
    args = ["s1", "walking", "t1", "--metadata", session_file(tmp_path),
            "--video", f"Cam0={cam0}", "--video", f"Cam1={cam1}"]
    result = cli(STAGE, *args, core_root=core)
    assert result.returncode == 0, result.stderr + result.stdout
    for cam in ("Cam0", "Cam1"):
        staged = core / "Data" / "s1" / "Videos" / cam / "InputMedia" / "walking" / "t1.mov"
        info = probe(staged)
        assert round(info["fps"]) == 30
        assert info["variable_frame_rate"] is False
    assert (core / "Data" / "s1" / "sessionMetadata.yaml").is_file()

    again = cli(STAGE, *args, core_root=core)
    assert again.returncode != 0
    assert "Refusing to overwrite" in again.stderr


def test_stage_refuses_missing_measurements(tmp_path, make_video, cli, write_intrinsics):
    core = core_with_profiles(tmp_path, write_intrinsics)
    cam0, cam1 = make_video("a.mp4"), make_video("b.mp4")
    result = cli(STAGE, "s1", "walking", "t1", "--metadata", session_file(tmp_path, height_m=...),
                 "--video", f"Cam0={cam0}", "--video", f"Cam1={cam1}", core_root=core)
    assert result.returncode != 0
    assert "height_m" in result.stderr
    assert not (core / "Data" / "s1").exists()


def test_stage_refuses_video_that_does_not_match_calibration(tmp_path, make_video, cli, write_intrinsics):
    core = core_with_profiles(tmp_path, write_intrinsics, size=(240, 320))  # calibrated in portrait
    cam0, cam1 = make_video("a.mp4"), make_video("b.mp4")
    result = cli(STAGE, "s1", "walking", "t1", "--metadata", session_file(tmp_path),
                 "--video", f"Cam0={cam0}", "--video", f"Cam1={cam1}", core_root=core)
    assert result.returncode != 0
    assert "different orientation" in result.stderr


def test_core_run_refuses_incomplete_session_before_heavy_imports(tmp_path, cli):
    core = tmp_path / "core"
    session = core / "Data" / "s1"
    session.mkdir(parents=True)
    (session / "sessionMetadata.yaml").write_text(yaml.safe_dump({"height_m": 1.3}))
    result = cli("medpose.engine.core_run", "s1", "walking", "t1", "--cameras", "Cam0", "Cam1", core_root=core)
    assert result.returncode != 0
    assert "mass_kg" in result.stderr


def fake_core_session(tmp_path, main_body="pass", metadata=None):
    """A stand-in Core checkout with a staged two-camera trial, to exercise the wrapper alone."""
    core = tmp_path / "core"
    session = core / "Data" / "s1"
    session.mkdir(parents=True)
    metadata = {"height_m": HEIGHT_M, "mass_kg": MASS_KG} if metadata is None else metadata
    (session / "sessionMetadata.yaml").write_text(yaml.safe_dump(metadata))
    for cam in CAMERAS:
        trial = session / "Videos" / cam / "InputMedia" / "walking"
        trial.mkdir(parents=True)
        (trial / "t1.mov").write_bytes(b"")
    (core / "utils.py").write_text("")
    (core / "main.py").write_text(f"def main(**kwargs):\n    {main_body}\n")
    return core


@pytest.mark.parametrize("error", ["OSError('disk full')", "Exception('bad calibration')"])
def test_core_failure_is_reported_with_nonzero_exit(tmp_path, cli, error):
    core = fake_core_session(tmp_path, f"raise {error}")
    result = cli(CORE_RUN, "s1", "walking", "t1", "--cameras", *CAMERAS, core_root=core)
    assert result.returncode == 1
    assert "OpenCap Core failed on s1/walking" in result.stderr
    assert "Don't use any output" in result.stderr
    assert "Traceback" in result.stderr
    assert "finished" not in result.stdout


def test_core_success_reports_output_locations(tmp_path, cli):
    core = fake_core_session(tmp_path)
    result = cli(CORE_RUN, "s1", "walking", "t1", "--cameras", *CAMERAS, core_root=core)
    assert result.returncode == 0, result.stderr
    assert "OpenCap Core finished" in result.stdout


@pytest.mark.parametrize("metadata, message", [
    ({}, "height_m"),
    ({"height_m": None, "mass_kg": MASS_KG}, "height_m"),
    ({"height_m": HEIGHT_M, "mass_kg": None}, "mass_kg"),
    ({"height_m": 130, "mass_kg": MASS_KG}, "units"),
])
def test_core_run_refuses_empty_null_or_implausible_metadata(tmp_path, cli, metadata, message):
    core = fake_core_session(tmp_path, "raise SystemExit('Core must not be reached')", metadata=metadata)
    result = cli(CORE_RUN, "s1", "walking", "t1", "--cameras", *CAMERAS, core_root=core)
    assert result.returncode != 0
    assert message in result.stderr
    assert "must not be reached" not in result.stderr


def test_core_run_refuses_unknown_session(tmp_path, cli):
    result = cli(CORE_RUN, "missing", "walking", "t1", "--cameras", *CAMERAS, core_root=tmp_path / "core")
    assert result.returncode != 0
    assert "prepared session" in result.stderr


def test_core_run_refuses_a_single_camera(tmp_path, cli):
    core = fake_core_session(tmp_path, "raise SystemExit('Core must not be reached')")
    result = cli(CORE_RUN, "s1", "walking", "t1", "--cameras", "Cam0", core_root=core)
    assert result.returncode != 0
    assert "at least two" in result.stderr


def test_core_run_refuses_a_camera_without_video(tmp_path, cli):
    core = fake_core_session(tmp_path, "raise SystemExit('Core must not be reached')")
    result = cli(CORE_RUN, "s1", "walking", "t1", "--cameras", "Cam0", "Cam7", core_root=core)
    assert result.returncode != 0
    assert "No video for Cam7" in result.stderr


def test_mono_refuses_incomplete_subject_before_touching_video(tmp_path, make_video, cli, write_intrinsics):
    mono = tmp_path / "mono"
    mono.mkdir()
    (mono / "run_mono_standalone.py").write_text("raise SystemExit('upstream must not be reached')\n")
    subject = tmp_path / "subject.yaml"
    subject.write_text(yaml.safe_dump({"height_m": 1.3, "mass_kg": 28.5}))  # no sex
    intrinsics = write_intrinsics(tmp_path / "cam.pickle", 320, 240)
    result = cli("medpose.engine.mono_run", make_video(), "--metadata", subject, "--intrinsics", intrinsics)
    assert result.returncode != 0
    assert "sex" in result.stderr
    assert not (tmp_path / "patient_data").exists()
