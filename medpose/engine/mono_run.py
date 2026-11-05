#!/usr/bin/env python3
"""Run OpenCap Monocular on one locally stored video and print its output paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from medpose.capture.video import check_intrinsics_match, normalize, probe
from medpose.paths import MONO_ROOT, NORMALIZED_ROOT
from medpose.subject import SubjectError, load_mono_subject


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process one video locally with OpenCap Monocular."
    )
    parser.add_argument("video", type=Path, help="Input camera video")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Subject YAML with height_m, mass_kg, and sex")
    parser.add_argument("--calibration", type=Path,
                        help="WHAM camera-motion calibration text file (required only for moving-camera mode)")
    parser.add_argument("--intrinsics", required=True, type=Path,
                        help="Camera intrinsics pickle calibrated for this Android lens/resolution")
    parser.add_argument("--activity", default="walking",
                        help="Activity label, e.g. walking, squats, or sts")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--rerun", action="store_true",
                        help="Recompute instead of reusing cached results")
    parser.add_argument("--estimate-world-camera", action="store_true",
                        help="Estimate camera motion; default assumes a fixed camera")
    args = parser.parse_args()

    if args.estimate_world_camera and args.calibration is None:
        parser.error("--calibration is required with --estimate-world-camera")
    for label, path in (("video", args.video), ("metadata", args.metadata),
                        ("intrinsics", args.intrinsics)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    if args.calibration is not None and not args.calibration.is_file():
        parser.error(f"calibration file does not exist: {args.calibration}")
    if not (MONO_ROOT / "run_mono_standalone.py").is_file():
        parser.error(f"OpenCap Monocular checkout not found at {MONO_ROOT}")

    # Upstream silently falls back to an adult default (1.70 m, 70 kg, male) for any missing
    # field, which is wrong for a child, so refuse incomplete metadata before it gets there.
    try:
        load_mono_subject(args.metadata)
    except SubjectError as e:
        parser.error(str(e))

    # Monocular reads orientation from Apple metadata only, so give it an upright,
    # constant-rate .mov whose size must match the Android intrinsics.
    video_path = NORMALIZED_ROOT / f"{args.video.stem}.mov"
    try:
        check_intrinsics_match(args.intrinsics, probe(args.video))
        # Monocular converts .mov to .avi only when no .avi exists; drop any stale copy.
        video_path.with_suffix(".avi").unlink(missing_ok=True)
        info = normalize(args.video.resolve(), video_path)
    except (OSError, RuntimeError, ValueError) as e:
        parser.error(str(e))
    print(f"Normalized {args.video} -> {video_path} ({info['width']}x{info['height']}, "
          f"{info['fps']:.0f} fps)")
    metadata_path = args.metadata.resolve()
    calibration_path = args.calibration.resolve() if args.calibration else None
    intrinsics_path = args.intrinsics.resolve()
    os.chdir(MONO_ROOT)
    sys.path.insert(0, str(MONO_ROOT))
    from run_mono_standalone import run_mono_standalone

    result = run_mono_standalone(
        video_path=str(video_path),
        metadata_path=str(metadata_path),
        calib_path=str(calibration_path) if calibration_path else "",
        intrinsics_path=str(intrinsics_path),
        estimate_local_only=not args.estimate_world_camera,
        rerun=args.rerun,
        session_id=args.session_id,
        activity=args.activity,
    )
    print(json.dumps(result, indent=2, default=str))
    return 1 if result.get("aborted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
