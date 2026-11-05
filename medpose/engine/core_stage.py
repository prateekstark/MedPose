#!/usr/bin/env python3
"""Normalize synchronized camera videos and stage them in OpenCap Core's layout.

Each video is written as an upright, constant-frame-rate .mov (see medpose/capture/video.py),
all at the same frame rate, so Core's iPhone-oriented loading code reads it correctly.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

from medpose.capture.video import check_intrinsics_match, nominal_fps, normalize, probe
from medpose.paths import CORE_ROOT
from medpose.subject import SubjectError, load_core_session


def parse_video(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use CAM=PATH, for example Cam0=/data/view0.mp4")
    cam, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not re.fullmatch(r"Cam\d+", cam) or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid camera name or missing video: {value}")
    return cam, path


def safe_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", value) and value not in {".", ".."})


def check_profiles(parser: argparse.ArgumentParser, metadata: Path, infos: dict) -> None:
    """Check each camera's intrinsics profile (iphoneModel in the metadata) fits its video."""
    try:
        with metadata.open() as f:
            models = (yaml.safe_load(f) or {}).get("iphoneModel") or {}
    except (OSError, yaml.YAMLError, AttributeError) as e:
        parser.error(f"Could not read iphoneModel from {metadata}: {e}")
    for cam, info in infos.items():
        model = models.get(cam)
        if model is None:
            parser.error(f"{metadata} has no iphoneModel entry for {cam}")
        profile = CORE_ROOT / "CameraIntrinsics" / model / "Deployed" / "cameraIntrinsics.pickle"
        if not profile.is_file():
            parser.error(f"No intrinsics profile for {cam} at {profile}; run "
                         f"python -m medpose.capture.intrinsics --core-camera-model {model} first")
        try:
            check_intrinsics_match(profile, info, label=f"{cam} video")
        except ValueError as e:
            parser.error(str(e))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage two or more synchronized videos for OpenCap Core."
    )
    parser.add_argument("session", help="New or existing local session name")
    parser.add_argument("trial", help="Trial folder name, e.g. walking")
    parser.add_argument("trial_id", help="Shared trial filename stem, e.g. walk01")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="OpenCap-format sessionMetadata.yaml")
    parser.add_argument("--video", required=True, action="append", type=parse_video,
                        metavar="CAM=PATH", help="Repeat for each Android camera")
    parser.add_argument("--fps", type=int,
                        help="Constant frame rate for all cameras (default: the slowest camera's rate)")
    args = parser.parse_args()

    if len(args.video) < 2:
        parser.error("Provide at least two camera videos")
    if not all(safe_name(value) for value in (args.session, args.trial, args.trial_id)):
        parser.error("Session, trial, and trial_id must be simple names without slashes")
    cams = [cam for cam, _ in args.video]
    if len(set(cams)) != len(cams):
        parser.error("Each camera name must appear only once")
    if not args.metadata.is_file():
        parser.error(f"Metadata file does not exist: {args.metadata}")
    try:
        load_core_session(args.metadata)
    except SubjectError as e:
        parser.error(str(e))

    session_dir = CORE_ROOT / "Data" / args.session
    metadata_target = session_dir / "sessionMetadata.yaml"
    try:
        differs = (metadata_target.exists()
                   and metadata_target.read_bytes() != args.metadata.resolve().read_bytes())
    except OSError as e:
        parser.error(f"Could not compare session metadata: {e}")
    if differs:
        parser.error(f"Existing session metadata differs; refusing to overwrite {metadata_target}")

    try:
        infos = {cam: probe(source) for cam, source in args.video}
    except RuntimeError as e:
        parser.error(str(e))
    rates = {cam: nominal_fps(info["fps"]) for cam, info in infos.items()}
    # Different phone models often record at different rates (e.g. 30 and 60 fps). Core
    # needs one shared rate, so faster videos are resampled down to the slowest one.
    fps = args.fps or min(rates.values())
    if args.fps is None and len(set(rates.values())) > 1:
        print(f"Cameras recorded at different frame rates {rates}; using {fps} fps for all "
              "(record every phone at the same rate to keep the faster detail)")

    check_profiles(parser, args.metadata, infos)

    targets = []
    for cam, source in args.video:
        target_dir = session_dir / "Videos" / cam / "InputMedia" / args.trial
        target = target_dir / f"{args.trial_id}.mov"
        # Core finds trial videos by "<trial_id>.*", so any existing match is a conflict.
        existing = list(target_dir.glob(args.trial_id + ".*"))
        if existing:
            parser.error(f"Refusing to overwrite existing video: {existing[0]}")
        targets.append((cam, source, target))

    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        if not metadata_target.exists():
            shutil.copy2(args.metadata.resolve(), metadata_target)
    except OSError as e:
        parser.error(f"Could not stage session metadata in {session_dir}: {e}")

    for cam, source, target in targets:
        if infos[cam]["variable_frame_rate"]:
            print(f"{cam}: variable frame rate detected; resampling to a constant {fps} fps")
        try:
            info = normalize(source, target, fps)
        except (OSError, RuntimeError, ValueError) as e:
            parser.error(f"{cam}: {e}")
        print(f"{cam}: {info['width']}x{info['height']} at {fps} fps -> {target}")

    print(f"Staged {len(args.video)} videos under {session_dir}")
    print("Check sessionMetadata.yaml and complete camera calibration before processing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
