#!/usr/bin/env python3
"""Calibrate a fixed multi-camera rig using a shared checkerboard."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", nargs="+", required=True, metavar="SOURCE")
    parser.add_argument("--pattern", nargs=2, type=int, default=(9, 6), metavar=("COLUMNS", "ROWS"),
                        help="Checkerboard inner corners; default: 9 6")
    parser.add_argument("--square-size", type=float, required=True,
                        help="Physical checkerboard square edge, e.g. 0.025 for 25 mm")
    parser.add_argument("--units", default="meters", help="Unit label for square size (default: meters)")
    parser.add_argument("--samples", type=int, default=15, help="Simultaneous board poses to capture")
    parser.add_argument("--output", type=Path, default=Path("calibration.json"))
    args = parser.parse_args()
    if len(args.cameras) < 2:
        parser.error("3D calibration needs at least two cameras")
    if args.square_size <= 0 or args.samples < 8 or min(args.pattern) < 2:
        parser.error("square size must be positive, samples at least 8, and pattern at least 2x2")

    captures = []
    for source in args.cameras:
        capture_source: int | str = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            for opened in captures:
                opened[1].release()
            raise SystemExit(f"Could not open camera source: {source}")
        captures.append((source, cap))

    columns, rows = args.pattern
    obj = np.zeros((columns * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    obj *= args.square_size
    observations: dict[str, list[np.ndarray]] = {source: [] for source, _ in captures}
    image_sizes: dict[str, tuple[int, int]] = {}
    object_points: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        while len(object_points) < args.samples:
            frames = []
            corners_by_camera = []
            for source, cap in captures:
                ok, frame = cap.read()
                if not ok:
                    frames.append(np.zeros((480, 640, 3), np.uint8))
                    corners_by_camera.append(None)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(
                    gray, (columns, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if found:
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(frame, (columns, rows), corners, found)
                cv2.putText(frame, f"Camera {source} | {'BOARD' if found else 'no board'}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 0) if found else (0, 0, 255), 2)
                frames.append(frame)
                corners_by_camera.append(corners if found else None)

            height = max(frame.shape[0] for frame in frames)
            width = max(frame.shape[1] for frame in frames)
            tiles = [cv2.resize(frame, (width, height)) for frame in frames]
            preview = cv2.vconcat([
                cv2.hconcat(tiles[i:i + 2]) if len(tiles[i:i + 2]) == 2 else
                cv2.hconcat(tiles[i:i + 2] + [np.zeros_like(tiles[i])])
                for i in range(0, len(tiles), 2)
            ])
            cv2.putText(preview, f"Samples {len(object_points)}/{args.samples} | SPACE capture | Q quit",
                        (15, preview.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("MedPose rig calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if all(corners is not None for corners in corners_by_camera):
                    object_points.append(obj.copy())
                    for (source, _), corners, frame in zip(captures, corners_by_camera, frames):
                        observations[source].append(corners)
                        image_sizes[source] = (frame.shape[1], frame.shape[0])
                    print(f"Captured calibration pose {len(object_points)}/{args.samples}")
                    time.sleep(0.2)
                else:
                    print("Show the checkerboard clearly in every camera before capturing.")
    finally:
        for _, cap in captures:
            cap.release()
        cv2.destroyAllWindows()

    if len(object_points) < 8:
        raise SystemExit("Need at least 8 shared checkerboard poses; calibration was not saved.")

    sizes = {}
    intrinsics = {}
    distortions = {}
    for source, _ in captures:
        image_size = image_sizes[source]
        sizes[source] = image_size
        _, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
            object_points, observations[source], image_size, None, None
        )
        intrinsics[source] = camera_matrix
        distortions[source] = distortion

    if len(set(sizes.values())) != 1:
        raise SystemExit("All cameras must use the same image resolution for this calibration workflow.")

    reference = args.cameras[0]
    cameras = {
        reference: {
            "image_size": sizes[reference],
            "camera_matrix": intrinsics[reference].tolist(),
            "distortion": distortions[reference].ravel().tolist(),
            "rotation_from_reference": np.eye(3).tolist(),
            "translation_from_reference": [0.0, 0.0, 0.0],
        }
    }
    for source in args.cameras[1:]:
        _, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
            object_points,
            observations[reference],
            observations[source],
            intrinsics[reference],
            distortions[reference],
            intrinsics[source],
            distortions[source],
            sizes[reference],
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        cameras[source] = {
            "image_size": sizes[source],
            "camera_matrix": intrinsics[source].tolist(),
            "distortion": distortions[source].ravel().tolist(),
            "rotation_from_reference": rotation.tolist(),
            "translation_from_reference": translation.ravel().tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "coordinate_frame": f"camera {reference}",
        "units": args.units,
        "cameras": cameras,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved multi-camera calibration to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
