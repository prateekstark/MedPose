"""Synthetic footage only: tests must never depend on real recordings."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.fixture
def make_video(tmp_path):
    def _make(name="clip.mp4", size=(320, 240), fps=30, seconds=1, extra=()):
        path = tmp_path / name
        ffmpeg("-f", "lavfi", "-i", f"testsrc=size={size[0]}x{size[1]}:rate={fps}:duration={seconds}",
               *extra, "-pix_fmt", "yuv420p", path)
        return path
    return _make


@pytest.fixture
def write_intrinsics():
    def _write(path, width, height):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"distortion": np.zeros((1, 5)), "intrinsicMat": np.eye(3),
                         "imageSize": np.array([[height], [width]], dtype=np.float64)}, f)
        return path
    return _write


@pytest.fixture
def cli(tmp_path):
    """Run a medpose module as a user would, against temporary project folders."""
    def _cli(module, *args, core_root=None):
        env = dict(os.environ, PYTHONPATH=str(ROOT),
                   MEDPOSE_PATIENT_ROOT=str(tmp_path / "patient_data"),
                   MEDPOSE_MONO_ROOT=str(tmp_path / "mono"),
                   MEDPOSE_CORE_ROOT=str(core_root or tmp_path / "core"))
        return subprocess.run([sys.executable, "-m", module, *map(str, args)],
                              capture_output=True, text=True, env=env, cwd=tmp_path)
    return _cli


def render_checkerboard_video(path, corners=(9, 6), size=(960, 720), fps=30, views=30, frames_per_view=2,
                              focal=800.0):
    """Render a flat board through a known pinhole camera at varied poses, including near the edges."""
    cols, rows = corners
    cell = 60
    board = np.full(((rows + 3) * cell, (cols + 3) * cell), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[(r + 1) * cell:(r + 2) * cell, (c + 1) * cell:(c + 2) * cell] = 0
    k = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    scale = 0.03 / cell  # 30 mm squares
    to_plane = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1.0]])
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    width_m, height_m = board.shape[1] * scale, board.shape[0] * scale
    for _ in range(views):
        rvec = rng.uniform(-0.5, 0.5, 3)
        rotation, _ = cv2.Rodrigues(rvec)
        depth = rng.uniform(0.45, 0.7)
        offset = rng.uniform(-0.18, 0.18, 2)
        t = np.array([offset[0], offset[1], depth]) - rotation @ np.array([width_m / 2, height_m / 2, 0])
        homography = k @ np.column_stack([rotation[:, 0], rotation[:, 1], t]) @ to_plane
        image = cv2.warpPerspective(board, homography, size, flags=cv2.INTER_AREA, borderValue=200)
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for _ in range(frames_per_view):
            writer.write(frame)
    writer.release()
    return path
