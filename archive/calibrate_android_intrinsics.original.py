#!/usr/bin/env python3
"""Calibrate one Android camera from a checkerboard video to OpenCap's pickle format."""

from __future__ import annotations

import argparse
import pickle
import tempfile
from pathlib import Path

import cv2
import numpy as np

from android_video import normalize


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate intrinsics from checkerboard footage recorded by one phone lens."
    )
    parser.add_argument("video", type=Path, help="Checkerboard video from the phone camera")
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
    if args.square_mm <= 0 or args.frame_step < 1 or args.max_views < 8:
        parser.error("Use a positive square size/frame step and at least 8 views")

    # Calibrate on the same upright, constant-rate frames that staging produces, so the
    # intrinsics match the normalized movement videos exactly.
    tmp_dir = tempfile.TemporaryDirectory()
    normalized = Path(tmp_dir.name) / "checkerboard.mov"
    try:
        normalize(args.video, normalized)
    except (RuntimeError, ValueError) as e:
        parser.error(str(e))
    capture = cv2.VideoCapture(str(normalized))
    if not capture.isOpened():
        parser.error(f"Could not read video: {args.video}")
    cols, rows = args.corners
    cell = args.square_mm / 1000.0
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * cell
    object_points, image_points = [], []
    image_size = None
    frame_index = 0
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while len(object_points) < args.max_views:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % args.frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            if found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                object_points.append(grid.copy())
                image_points.append(refined)
        frame_index += 1
    capture.release()
    tmp_dir.cleanup()

    if len(object_points) < 8:
        parser.error(f"Found only {len(object_points)} usable board views; need at least 8")
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    params = {
        "distortion": distortion,
        "intrinsicMat": camera_matrix,
        # OpenCap stores [[height], [width]], matching its bundled iPhone profiles.
        "imageSize": np.array([[image_size[1]], [image_size[0]]], dtype=np.float64),
    }
    outputs = []
    if args.output is not None:
        outputs.append(args.output.expanduser().resolve())
    if args.core_camera_model:
        outputs.append(
            ROOT / "opencap-core" / "CameraIntrinsics" / args.core_camera_model
            / "Deployed" / "cameraIntrinsics.pickle"
        )
    for output in outputs:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("wb") as f:
                pickle.dump(params, f)
        except OSError as e:
            parser.error(f"Could not write {output}: {e}")
        print(f"Saved {output}")
    print(f"RMS reprojection error: {rms:.4f}px using {len(object_points)} views at {image_size[0]}x{image_size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
