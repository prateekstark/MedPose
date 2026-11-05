"""Fails if a change would let patient data reach git or Claude's file tools."""

import json
import subprocess

import pytest

from conftest import ROOT


@pytest.mark.parametrize("path", [
    "patient_data/session1/walk.mp4",
    "patient_data/notes.txt",
    "normalized/walk.mov",
    "anywhere/clip.MOV",
    "anywhere/clip.webm",
    "anywhere/cameraIntrinsics.pickle",
    "opencap-core/Data/s1/sessionMetadata.yaml",
    "opencap-monocular/results/out.json",
])
def test_git_ignores_patient_data(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert result.returncode == 0, f"{path} isn't ignored"


def test_claude_is_denied_patient_folders():
    deny = json.loads((ROOT / ".claude" / "settings.json").read_text())["permissions"]["deny"]
    for rule in ("Read(./patient_data/**)", "Read(./normalized/**)",
                 "Read(./opencap-core/Data/**)", "Read(./opencap-monocular/results/**)"):
        assert rule in deny
