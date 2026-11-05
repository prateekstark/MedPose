#!/usr/bin/env python3
"""Calibrate one camera (Android phone or webcam) from a checkerboard video.

Writes OpenCap's intrinsics pickle. A bad calibration gives wrong joint angles with no visible
error, so the result must pass a quality gate before anything is saved.
"""

from __future__ import annotations

import argparse
import pickle
import tempfile
from pathlib import Path

import cv2
import numpy as np

from medpose.capture.video import normalize
from medpose.paths import CORE_ROOT

MIN_VIEWS = 8
# Rules of thumb, not validated limits: well-captured boards usually stay under 1 px, and
# boards seen only near the centre leave lens distortion poorly constrained at the edges.
DEFAULT_MAX_RMS_PX = 1.0
DEFAULT_MIN_COVERAGE = 0.5
COVERAGE_GRID = (6, 4)  # image cells across, down


def detect_board(video: Path, corners: tuple, square_mm: float, frame_step: int, max_views: int):
    """Return (object_points, image_points, image_size) from every usable sampled frame."""
    cols, rows = corners
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_mm / 1000.0)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not read video: {video}")
    object_points, image_points = [], []
    image_size = None
    frame_index = 0
    while len(object_points) < max_views:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            found, found_corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            if found:
                image_points.append(cv2.cornerSubPix(gray, found_corners, (11, 11), (-1, -1), criteria))
                object_points.append(grid.copy())
        frame_index += 1
    capture.release()
    return object_points, image_points, image_size


def image_coverage(image_points: list, image_size: tuple) -> float:
    """Fraction of image grid cells that contain at least one detected corner."""
    if not image_points:
        return 0.0
    width, height = image_size
    cells_x, cells_y = COVERAGE_GRID
    points = np.concatenate([p.reshape(-1, 2) for p in image_points])
    cx = np.clip((points[:, 0] / width * cells_x).astype(int), 0, cells_x - 1)
    cy = np.clip((points[:, 1] / height * cells_y).astype(int), 0, cells_y - 1)
    return len(set(zip(cx.tolist(), cy.tolist()))) / (cells_x * cells_y)


def quality_problems(rms: float, views: int, coverage: float,
                     max_rms: float = DEFAULT_MAX_RMS_PX,
                     min_coverage: float = DEFAULT_MIN_COVERAGE) -> list:
    """Return a description of each failed check; an empty list means it passed."""
    problems = []
    if views < MIN_VIEWS:
        problems.append(f"only {views} usable board views; need at least {MIN_VIEWS}")
    if rms > max_rms:
        problems.append(f"reprojection error {rms:.2f}px is above {max_rms:.2f}px; "
                        "hold the board still and in focus, with stabilisation and autofocus off")
    if coverage < min_coverage:
        problems.append(f"the board covered {coverage:.0%} of the image, below {min_coverage:.0%}; "
                        "move it into the corners and edges, at varied angles")
    return problems


def intrinsics_params(camera_matrix: np.ndarray, distortion: np.ndarray, image_size: tuple) -> dict:
    # OpenCap stores [[height], [width]], matching its bundled iPhone profiles.
    return {"distortion": distortion, "intrinsicMat": camera_matrix,
            "imageSize": np.array([[image_size[1]], [image_size[0]]], dtype=np.float64)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate intrinsics from checkerboard footage recorded by one camera lens."
    )
    parser.add_argument("video", type=Path, help="Checkerboard video from the camera")
    parser.add_argument("--corners", nargs=2, type=int, required=True,
                        metavar=("COLUMNS", "ROWS"),
                        help="Inner corner count across then down, e.g. 9 6")
    parser.add_argument("--square-mm", type=float, required=True,
                        help="Measured checkerboard square edge in millimeters")
    parser.add_argument("--output", type=Path,
                        help="Optional output cameraIntrinsics.pickle path")
    parser.add_argument("--core-camera-model",
                        help="Also install under OpenCap Core/CameraIntrinsics/<name>/Deployed")
    parser.add_argument("--frame-step", type=int, default=15,
                        help="Sample every Nth frame (default: 15)")
    parser.add_argument("--max-views", type=int, default=50,
                        help="Maximum accepted checkerboard views (default: 50)")
    parser.add_argument("--max-rms", type=float, default=DEFAULT_MAX_RMS_PX,
                        help=f"Largest accepted reprojection error in pixels (default: {DEFAULT_MAX_RMS_PX})")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help=f"Smallest accepted image coverage, 0-1 (default: {DEFAULT_MIN_COVERAGE})")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if args.output is None and not args.core_camera_model:
        parser.error("Provide --output, --core-camera-model, or both")
    if args.core_camera_model and (
        "/" in args.core_camera_model or "\\" in args.core_camera_model
        or args.core_camera_model in {".", ".."}
    ):
        parser.error("--core-camera-model must be a simple camera-model label")
    if args.square_mm <= 0 or args.frame_step < 1 or args.max_views < MIN_VIEWS:
        parser.error(f"Use a positive square size/frame step and at least {MIN_VIEWS} views")

    # Calibrate on the same upright, constant-rate frames that staging produces, so the
    # intrinsics match the normalized movement videos exactly.
    with tempfile.TemporaryDirectory() as tmp_dir:
        normalized = Path(tmp_dir) / "checkerboard.mov"
        try:
            normalize(args.video, normalized)
            object_points, image_points, image_size = detect_board(
                normalized, tuple(args.corners), args.square_mm, args.frame_step, args.max_views)
        except (RuntimeError, ValueError) as e:
            parser.error(str(e))

    if len(object_points) < MIN_VIEWS:
        parser.error(f"Found only {len(object_points)} usable board views; need at least {MIN_VIEWS}")
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    coverage = image_coverage(image_points, image_size)
    print(f"RMS reprojection error: {rms:.4f}px using {len(object_points)} views at "
          f"{image_size[0]}x{image_size[1]}; board covered {coverage:.0%} of the image")
    problems = quality_problems(rms, len(object_points), coverage, args.max_rms, args.min_coverage)
    if problems:
        parser.exit(1, "Calibration rejected, nothing saved:\n" + "".join(f"- {p}\n" for p in problems))

    params = intrinsics_params(camera_matrix, distortion, image_size)
    outputs = []
    if args.output is not None:
        outputs.append(args.output.expanduser().resolve())
    if args.core_camera_model:
        outputs.append(CORE_ROOT / "CameraIntrinsics" / args.core_camera_model
                       / "Deployed" / "cameraIntrinsics.pickle")
    for output in outputs:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("wb") as f:
                pickle.dump(params, f)
        except OSError as e:
            parser.error(f"Could not write {output}: {e}")
        print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
