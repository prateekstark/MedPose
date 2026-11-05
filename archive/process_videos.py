#!/usr/bin/env python3
"""Process synchronized multi-camera videos into a 3D skeleton video and JSONL."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medpose import (
    draw_3d_skeleton,
    draw_pose,
    load_calibration,
    mp_pose,
    triangulate_landmarks,
)


def parse_camera(value: str) -> tuple[str, Path]:
    camera_id, separator, path = value.partition("=")
    if not separator or not camera_id or not path:
        raise argparse.ArgumentTypeError("camera input must be CAMERA_ID=VIDEO_PATH")
    return camera_id, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="append", type=parse_camera, required=True,
                        metavar="ID=VIDEO", help="Repeat once per synchronized view")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, default=Path("medpose_3d_demo.mp4"))
    parser.add_argument("--jsonl-output", type=Path, default=Path("medpose_3d.jsonl"))
    parser.add_argument("--min-visibility", type=float, default=0.4)
    args = parser.parse_args()
    ids = [camera_id for camera_id, _ in args.camera]
    paths = [path for _, path in args.camera]
    if len(ids) < 2:
        parser.error("3D reconstruction requires at least two camera videos")
    if len(set(ids)) != len(ids):
        parser.error("camera IDs must be unique")
    if not 0 <= args.min_visibility <= 1:
        parser.error("min visibility must be between 0 and 1")

    calibration, units = load_calibration(args.calibration, ids)
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise SystemExit("Could not open every video. Check the camera paths.")

    fps_values = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
    dimensions = [
        (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for capture in captures
    ]
    writer = None
    frame_index = 0
    try:
        if min(fps_values) <= 0 or max(fps_values) - min(fps_values) > 0.01:
            raise SystemExit("Videos must have matching, constant frame rates.")
        if len(set(dimensions)) != 1:
            raise SystemExit("Videos must use a matching resolution in this first version.")
        for camera_id, (width, height) in zip(ids, dimensions):
            if (width, height) != calibration[camera_id]["size"]:
                raise SystemExit(
                    f"Video for {camera_id} is {width}x{height}; calibration expects "
                    f"{calibration[camera_id]['size'][0]}x{calibration[camera_id]['size'][1]}"
                )

        args.video_output.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        view_size = (480, 270)
        panel_3d_size = (648, 540)
        canvas_size = (1608, 540)
        writer = cv2.VideoWriter(
            str(args.video_output), cv2.VideoWriter_fourcc(*"mp4v"),
            fps_values[0], canvas_size,
        )
        if not writer.isOpened():
            raise SystemExit(f"Could not create output video: {args.video_output}")

        with args.jsonl_output.open("w", encoding="utf-8") as output, ExitStack() as stack:
            trackers = {
                camera_id: stack.enter_context(
                    mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                )
                for camera_id in ids
            }
            while True:
                frames: dict[str, Any] = {}
                for camera_id, capture in zip(ids, captures):
                    ok, frame = capture.read()
                    if not ok:
                        frames = {}
                        break
                    frames[camera_id] = frame
                if not frames:
                    break

                results = {}
                views = []
                for camera_id in ids:
                    frame = frames[camera_id]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results[camera_id] = trackers[camera_id].process(rgb)
                    view = draw_pose(frame.copy(), results[camera_id].pose_landmarks)
                    view = cv2.resize(view, view_size)
                    cv2.putText(view, camera_id, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 255, 255), 2)
                    views.append(view)

                joints = triangulate_landmarks(ids, results, frames, calibration, args.min_visibility)
                canvas_views = np.zeros((540, 960, 3), dtype=np.uint8)
                for index, view in enumerate(views[:4]):
                    row, column = divmod(index, 2)
                    y, x = row * view_size[1], column * view_size[0]
                    canvas_views[y:y + view_size[1], x:x + view_size[0]] = view
                panel_3d = cv2.resize(draw_3d_skeleton(joints, units), panel_3d_size)
                writer.write(cv2.hconcat((canvas_views, panel_3d)))
                output.write(json.dumps({
                    "frame": frame_index,
                    "time_seconds": round(frame_index / fps_values[0], 6),
                    "coordinate_frame": f"camera {ids[0]}",
                    "units": units,
                    "joints_3d": joints,
                }) + "\n")
                frame_index += 1
    finally:
        if writer is not None:
            writer.release()
        for capture in captures:
            capture.release()

    if frame_index == 0:
        raise SystemExit("No frames were processed.")
    print(f"Processed {frame_index} synchronized frames at {fps_values[0]:.3f} fps.")
    print(f"3D preview: {args.video_output}")
    print(f"3D coordinates: {args.jsonl_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
