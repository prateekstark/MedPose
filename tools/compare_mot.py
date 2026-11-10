#!/usr/bin/env python3
"""Compare two OpenSim .mot joint-angle files and print only summary numbers.

Frames are matched by timestamp, not by row number: the second file is linearly
interpolated onto the first file's times, and times outside its range or across
gaps longer than two of its frames are skipped. Files recorded at different rates
or start times therefore compare correctly; an unknown clock offset between two
systems still needs ``python -m benchmarks.compare``.

Prints no patient details, so its output is safe to share when asking for help.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.align import resample  # noqa: E402

# Translations are in metres and everything else in degrees, so they're summarised separately.
TRANSLATIONS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}


def read_mot(path: Path, with_time: bool = False) -> tuple:
    """Return (column names without time, values with shape frames x coordinates[, time])."""
    lines = Path(path).read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader") + 1
    except StopIteration:
        raise ValueError(f"{path} has no endheader line, so it isn't a .mot file") from None
    names = lines[start].split()
    if not names or names[0] != "time":
        raise ValueError(f"{path}: expected a time column first, got {names[:1]}")
    rows = [line for line in lines[start + 1:] if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no data rows")
    values = np.atleast_2d(np.loadtxt(rows))
    if with_time:
        return names[1:], values[:, 1:], values[:, 0]
    return names[1:], values[:, 1:]


def compare(first: Path, second: Path) -> dict:
    """Return per-coordinate RMS difference at the first file's timestamps."""
    names_a, a, time_a = read_mot(first, with_time=True)
    names_b, b, time_b = read_mot(second, with_time=True)
    if names_a != names_b:
        raise ValueError("The two files have different coordinates")
    if len(time_b) == len(time_a) and np.allclose(time_a, time_b):
        b_on_a = b
    elif len(time_b) < 2:
        raise ValueError("The second file needs at least two frames to match timestamps")
    else:
        b_on_a = resample(time_b, b, time_a, max_gap_s=2.0 * float(np.median(np.diff(time_b))) + 1e-9)
    difference = a - b_on_a
    compared = np.isfinite(difference).all(axis=1)
    if not compared.any():
        raise ValueError("The files share no time range")
    rms = np.sqrt(np.mean(difference[compared] ** 2, axis=0))
    return {"frames_a": len(a), "frames_b": len(b), "frames_compared": int(compared.sum()),
            "rms": dict(zip(names_a, rms.tolist()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.first, args.second)
    except (OSError, ValueError) as e:
        parser.exit(1, f"Error: {e}\n")
    rms = result["rms"]
    rotations = {k: v for k, v in rms.items() if k not in TRANSLATIONS}
    worst = max(rotations, key=rotations.get)
    print(f"frames: {result['frames_a']} vs {result['frames_b']}; compared at {result['frames_compared']} shared timestamps")
    print(f"rotations: mean RMS {np.mean(list(rotations.values())):.3f} deg, "
          f"worst {worst} {rotations[worst]:.3f} deg")
    print("translations RMS: " + ", ".join(f"{k} {rms[k] * 1000:.1f} mm" for k in sorted(TRANSLATIONS) if k in rms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
