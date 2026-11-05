#!/usr/bin/env python3
"""Normalize Android phone videos so OpenCap Core and Monocular read them correctly.

OpenCap expects iPhone recordings: a .mov file, constant frame rate, and orientation
in Apple's `com.apple.quicktime.video-orientation` tag. Videos without that tag are
treated as upright, with intrinsics used exactly as calibrated. Android phones write
.mp4 files, often at a variable frame rate, with orientation in a display matrix.

Normalizing bakes the orientation into the pixels, resamples to a constant frame rate,
and writes a .mov without rotation metadata. Calibrating intrinsics from a normalized
checkerboard video then gives intrinsics that match the normalized movement videos.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} was not found on PATH; install ffmpeg first")


def probe(video: Path) -> dict:
    """Return displayed size, nominal frame rate, rotation, and whether timing is variable."""
    _require_ffmpeg()
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
             "-show_entries", "packet=pts_time", "-of", "json", str(video)],
            capture_output=True, text=True,
        )
    except OSError as e:
        raise RuntimeError(f"Could not run ffprobe on {video}: {e}") from e
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {video}: {result.stderr.strip()}")
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned unreadable output for {video}: {e}") from e
    if not info.get("streams"):
        raise RuntimeError(f"No video stream in {video}")
    stream = info["streams"][0]

    rotation = 0
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(round(float(side_data["rotation"])))
    if "rotate" in stream.get("tags", {}):  # older muxers
        rotation = -int(stream["tags"]["rotate"])
    width, height = int(stream["width"]), int(stream["height"])
    if rotation % 180:
        width, height = height, width

    times = np.sort([float(p["pts_time"]) for p in info.get("packets", [])
                     if p.get("pts_time") not in (None, "N/A")])
    intervals = np.diff(times)
    if len(intervals):
        median = float(np.median(intervals))
        # Average the regular intervals (gaps from dropped frames excluded): containers
        # like .mkv round timestamps to whole milliseconds, which skews the median.
        regular = intervals[intervals <= 1.5 * median]
        fps = 1.0 / float(regular.mean()) if median > 0 else float(Fraction(stream["r_frame_rate"]))
        # Constant-rate video has (almost) identical frame intervals.
        variable = bool(np.any(np.abs(intervals - median) > 0.25 * median))
    else:
        fps, variable = float(Fraction(stream["r_frame_rate"])), False

    return {"width": width, "height": height, "rotation": rotation,
            "fps": fps, "variable_frame_rate": variable, "frames": len(times)}


KEYFRAMES_PER_S = 10


def nominal_fps(fps: float) -> int:
    """Snap a measured rate (e.g. 29.97 or 59.4) to the nearest whole number."""
    return max(1, int(round(fps)))


def normalize(source: Path, target: Path, fps: int | None = None) -> dict:
    """Write an upright, constant-frame-rate .mov copy of source and return its probe."""
    if target.suffix.lower() != ".mov":
        raise ValueError(f"Normalized videos must be .mov for OpenCap: {target}")
    before = probe(source)
    fps = fps or nominal_fps(before["fps"])
    # ffmpeg applies the display-matrix rotation by default (autorotate) and does not
    # copy it to the output. The fps filter drops/duplicates frames onto a fixed grid.
    # Core grabs calibration and neutral-pose stills by seeking to keyframes near given
    # times, so keyframes must be frequent (iPhones write them often; x264's default
    # of one per 250 frames leaves short clips with a single keyframe at the start).
    keyint = str(max(1, round(fps / KEYFRAMES_PER_S)))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
             "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "medium", "-crf", "12",
             "-g", keyint, "-keyint_min", keyint,
             "-pix_fmt", "yuv420p", "-an", "-map_metadata", "-1", str(target)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            target.unlink(missing_ok=True)
    except OSError as e:
        raise RuntimeError(f"Could not write {target}: {e}") from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not normalize {source}: {result.stderr.strip()[-500:]}")
    after = probe(target)
    if after["rotation"] != 0 or (after["width"], after["height"]) != (before["width"], before["height"]):
        raise RuntimeError(f"Normalization of {source} did not produce an upright video: {after}")
    return after


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Load numpy data only, so a tampered intrinsics file can't run code."""

    ALLOWED = {
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from an intrinsics file")
        return super().find_class(module, name)


def load_intrinsics(intrinsics: Path) -> dict:
    """Read an OpenCap intrinsics pickle, allowing numpy data only."""
    with intrinsics.open("rb") as f:
        return _NumpyOnlyUnpickler(f).load()


def intrinsics_size(intrinsics: Path) -> tuple[int, int]:
    """Return (width, height) from an OpenCap intrinsics pickle ([[height], [width]])."""
    try:
        params = load_intrinsics(intrinsics)
        size = np.asarray(params["imageSize"]).flatten()
        return int(size[1]), int(size[0])
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Could not read the image size from intrinsics {intrinsics}: {e}") from e


def check_intrinsics_match(intrinsics: Path, video_info: dict, label: str = "video") -> None:
    """Refuse a video whose upright size differs from the calibrated size.

    OpenCap would otherwise rotate or rescale the intrinsics on the assumption that they
    came from a portrait iPhone calibration, which is wrong for Android profiles.
    """
    calibrated = intrinsics_size(intrinsics)
    actual = (video_info["width"], video_info["height"])
    if calibrated != actual:
        hint = (" The phone was held in a different orientation than during calibration."
                if calibrated == actual[::-1] else
                " Record calibration and movement with the same lens and resolution.")
        raise ValueError(f"{label} is {actual[0]}x{actual[1]} but {intrinsics} was calibrated "
                         f"at {calibrated[0]}x{calibrated[1]}.{hint}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an Android video, or write an OpenCap-ready .mov copy."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="Optional output .mov path")
    parser.add_argument("--fps", type=int, help="Constant output frame rate (default: nominal rate)")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    try:
        print(json.dumps(probe(args.video), indent=2))
        if args.output:
            print(json.dumps(normalize(args.video, args.output, args.fps), indent=2))
            print(f"Saved {args.output}")
    except (RuntimeError, ValueError) as e:
        parser.exit(1, f"Error: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
