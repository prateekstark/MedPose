#!/usr/bin/env python3
"""Live, per-camera human pose tracking with an optional JSONL landmark stream."""

from __future__ import annotations

import argparse
import json
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class CameraFrame:
    source: str
    frame: Any | None = None
    sequence: int = 0
    captured_at: float = 0.0
    error: str | None = None


class CameraReader:
    """Continuously read a camera so a slow inference step does not stall capture."""

    def __init__(self, source: str):
        self.source = source
        camera_arg: int | str = int(source) if source.isdigit() else source
        self.capture = cv2.VideoCapture(camera_arg)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self.state = CameraFrame(source=source)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        while not self.stopped.is_set():
            ok, frame = self.capture.read()
            if not ok:
                with self.lock:
                    self.state.error = "No frame received"
                time.sleep(0.05)
                continue
            with self.lock:
                self.state.frame = frame
                self.state.sequence += 1
                self.state.captured_at = time.time()
                self.state.error = None

    def snapshot(self) -> CameraFrame:
        with self.lock:
            return CameraFrame(**self.state.__dict__)

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=1)
        self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track human pose from one or more live cameras."
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["0"],
        metavar="SOURCE",
        help="Camera index or stream URL, e.g. --cameras 0 1 or --cameras 0 rtsp://...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poses.jsonl"),
        help="Write per-frame landmarks as JSON Lines (default: poses.jsonl).",
    )
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Calibration JSON written by calibrate.py")
    parser.add_argument("--sync-tolerance-ms", type=float, default=50,
                        help="Maximum timestamp difference across cameras for a 3D frame")
    parser.add_argument("--min-visibility", type=float, default=0.4,
                        help="Minimum landmark visibility used for 3D triangulation")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def draw_pose(frame: Any, landmarks: Any | None) -> Any:
    if landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    return frame


def load_calibration(path: Path, sources: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = data.get("cameras", {})
    if len(sources) < 2:
        raise SystemExit("3D reconstruction requires at least two cameras.")
    missing = [source for source in sources if source not in cameras]
    if missing:
        raise SystemExit(f"Calibration is missing camera source(s): {', '.join(missing)}")
    result: dict[str, dict[str, np.ndarray]] = {}
    for source in sources:
        camera = cameras[source]
        result[source] = {
            "K": np.asarray(camera["camera_matrix"], dtype=np.float64),
            "dist": np.asarray(camera["distortion"], dtype=np.float64),
            "size": tuple(camera["image_size"]),
            "R": np.asarray(camera["rotation_from_reference"], dtype=np.float64),
            "T": np.asarray(camera["translation_from_reference"], dtype=np.float64).reshape(3, 1),
        }
    return result, data.get("units", "calibration units")


def triangulate_landmarks(
    sources: list[str], results: dict[str, Any], frames: dict[str, Any],
    calibration: dict[str, dict[str, np.ndarray]], min_visibility: float,
) -> list[dict[str, Any] | None]:
    """Triangulate each corresponding body joint into the reference camera frame."""
    reference = sources[0]
    reference_pose = results[reference].pose_landmarks if results.get(reference) else None
    if reference_pose is None:
        return [None] * 33

    projections = {}
    normalized_points = {}
    visibility = {}
    for source in sources:
        result = results.get(source)
        pose_landmarks = result.pose_landmarks if result else None
        if pose_landmarks is None:
            continue
        height, width = frames[source].shape[:2]
        camera = calibration[source]
        rotation, translation = camera["R"], camera["T"]
        projections[source] = np.hstack((rotation, translation))
        points = []
        scores = []
        for point in pose_landmarks.landmark:
            pixel = np.asarray([[[point.x * width, point.y * height]]], dtype=np.float64)
            undistorted = cv2.undistortPoints(pixel, camera["K"], camera["dist"])[0, 0]
            points.append(undistorted)
            scores.append(float(point.visibility))
        normalized_points[source] = np.asarray(points)
        visibility[source] = scores

    output: list[dict[str, Any] | None] = []
    names = [mp_pose.PoseLandmark(i).name.lower() for i in range(33)]
    for joint_index, name in enumerate(names):
        views = [
            source for source in sources
            if source in normalized_points and visibility[source][joint_index] >= min_visibility
        ]
        if len(views) < 2:
            output.append(None)
            continue
        rows = []
        for source in views:
            x, y = normalized_points[source][joint_index]
            projection = projections[source]
            rows.append(x * projection[2] - projection[0])
            rows.append(y * projection[2] - projection[1])
        _, _, vt = np.linalg.svd(np.asarray(rows))
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-10:
            output.append(None)
            continue
        xyz = homogeneous[:3] / homogeneous[3]
        output.append({
            "name": name,
            "x": round(float(xyz[0]), 6),
            "y": round(float(xyz[1]), 6),
            "z": round(float(xyz[2]), 6),
            "visibility": round(float(min(visibility[source][joint_index] for source in views)), 6),
        })
    return output


def draw_3d_skeleton(joints: list[dict[str, Any] | None], units: str) -> Any:
    canvas = np.zeros((600, 720, 3), dtype=np.uint8)
    canvas[:] = (22, 25, 31)
    points = {j["name"]: np.array([j["x"], j["y"], j["z"]]) for j in joints if j is not None}
    if not points:
        cv2.putText(canvas, "Waiting for synchronized poses", (35, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        return canvas
    values = np.stack(list(points.values()))
    center = np.median(values, axis=0)
    relative = values - center
    # Axonometric view: show lateral, vertical and depth motion at once.
    projected = {name: np.array([p[0] - 0.55 * p[2], p[1] - 0.3 * p[2]])
                 for name, p in zip(points, relative)}
    extent = max(float(np.ptp(np.stack(list(projected.values())), axis=0).max()), 0.1)
    scale = 420.0 / extent
    screen = {
        name: (int(360 + p[0] * scale), int(300 + p[1] * scale))
        for name, p in projected.items()
    }
    for first, second in mp_pose.POSE_CONNECTIONS:
        first_name = mp_pose.PoseLandmark(first).name.lower()
        second_name = mp_pose.PoseLandmark(second).name.lower()
        if first_name in screen and second_name in screen:
            cv2.line(canvas, screen[first_name], screen[second_name], (60, 205, 255), 4, cv2.LINE_AA)
    for name, point in screen.items():
        cv2.circle(canvas, point, 6, (70, 245, 150), -1, cv2.LINE_AA)
    cv2.putText(canvas, f"3D pose | reference camera frame | {units}", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 240, 245), 2)
    return canvas


def main() -> int:
    args = parse_args()
    if not 0 <= args.min_detection_confidence <= 1:
        raise SystemExit("min detection confidence must be between 0 and 1")
    if not 0 <= args.min_tracking_confidence <= 1:
        raise SystemExit("min tracking confidence must be between 0 and 1")
    if not 0 <= args.min_visibility <= 1 or args.sync_tolerance_ms <= 0:
        raise SystemExit("min visibility must be in [0, 1] and sync tolerance must be positive")

    calibration, units = load_calibration(args.calibration, args.cameras)

    readers: list[CameraReader] = []
    try:
        for source in args.cameras:
            readers.append(CameraReader(source))
        for reader in readers:
            snapshot = reader.snapshot()
            deadline = time.time() + 5
            while snapshot.frame is None and time.time() < deadline:
                time.sleep(0.05)
                snapshot = reader.snapshot()
            if snapshot.frame is None:
                raise RuntimeError(f"Camera {reader.source} did not provide a frame")
            actual_size = (snapshot.frame.shape[1], snapshot.frame.shape[0])
            if actual_size != calibration[reader.source]["size"]:
                raise RuntimeError(
                    f"Camera {reader.source} is {actual_size[0]}x{actual_size[1]}, but calibration "
                    f"expects {calibration[reader.source]['size'][0]}x{calibration[reader.source]['size'][1]}"
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output, ExitStack() as stack:
            poses = {
                reader.source: stack.enter_context(
                    mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=args.model_complexity,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=args.min_detection_confidence,
                        min_tracking_confidence=args.min_tracking_confidence,
                    )
                )
                for reader in readers
            }
            last_sequence = {reader.source: -1 for reader in readers}
            last_written_key = None
            latest_results: dict[str, Any] = {}
            latest_frames: dict[str, Any] = {}
            while True:
                snapshots = {reader.source: reader.snapshot() for reader in readers}
                ready = all(s.frame is not None for s in snapshots.values())
                synchronized = ready and (
                    max(s.captured_at for s in snapshots.values()) -
                    min(s.captured_at for s in snapshots.values())
                    <= args.sync_tolerance_ms / 1000.0
                )
                if synchronized:
                    for reader in readers:
                        snapshot = snapshots[reader.source]
                        if snapshot.sequence == last_sequence[snapshot.source]:
                            continue
                        rgb = cv2.cvtColor(snapshot.frame, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        latest_results[snapshot.source] = poses[snapshot.source].process(rgb)
                        latest_frames[snapshot.source] = snapshot.frame.copy()
                        last_sequence[snapshot.source] = snapshot.sequence

                # Processed frames carry their capture timestamp; only emit once every
                # camera has contributed a new synchronized observation.
                frame_key = tuple(last_sequence[s] for s in args.cameras)
                if synchronized and len(latest_results) == len(readers) and frame_key != last_written_key:
                    joints_3d = triangulate_landmarks(
                        args.cameras, latest_results, latest_frames, calibration, args.min_visibility
                    )
                    timestamp = datetime.fromtimestamp(
                        min(snapshots[s].captured_at for s in args.cameras), timezone.utc
                    ).isoformat()
                    output.write(json.dumps({
                        "timestamp": timestamp,
                        "coordinate_frame": f"camera {args.cameras[0]}",
                        "units": units,
                        "joints_3d": joints_3d,
                    }) + "\n")
                    output.flush()
                    last_written_key = frame_key

                tiles = []
                for reader in readers:
                    snapshot = snapshots[reader.source]
                    if snapshot.frame is None:
                        tile = np.zeros((360, 640, 3), dtype="uint8")
                        cv2.putText(
                            tile,
                            f"Camera {snapshot.source}: {snapshot.error or 'waiting'}",
                            (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        tiles.append(tile)
                        continue

                    frame = snapshot.frame.copy()
                    result = latest_results.get(snapshot.source)
                    if result:
                        draw_pose(frame, result.pose_landmarks)
                    cv2.putText(
                        frame,
                        f"Camera {snapshot.source}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    tiles.append(frame)

                if tiles:
                    width = max(tile.shape[1] for tile in tiles)
                    height = max(tile.shape[0] for tile in tiles)
                    normalized = [cv2.resize(tile, (width, height)) for tile in tiles]
                    columns = min(2, len(normalized))
                    rows = (len(normalized) + columns - 1) // columns
                    blank = np.zeros((height, width, 3), dtype="uint8")
                    while len(normalized) < rows * columns:
                        normalized.append(blank.copy())
                    grid_rows = [
                        cv2.hconcat(normalized[start : start + columns])
                        for start in range(0, len(normalized), columns)
                    ]
                    cv2.imshow("MedPose - live pose tracking", cv2.vconcat(grid_rows))
                if len(latest_results) == len(readers):
                    current_3d = triangulate_landmarks(
                        args.cameras, latest_results, latest_frames, calibration, args.min_visibility
                    )
                    cv2.imshow("MedPose - reconstructed 3D pose", draw_3d_skeleton(current_3d, units))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for reader in readers:
            reader.close()
        cv2.destroyAllWindows()
    return 0


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


if __name__ == "__main__":
    raise SystemExit(main())
