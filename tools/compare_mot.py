#!/usr/bin/env python3
"""Compare two OpenSim .mot joint-angle files and print only summary numbers.

Prints no patient details, so its output is safe to share when asking for help.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Translations are in metres and everything else in degrees, so they're summarised separately.
TRANSLATIONS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}


def read_mot(path: Path) -> tuple:
    """Return (column names without time, values with shape frames x coordinates)."""
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
    return names[1:], values[:, 1:]


def compare(first: Path, second: Path) -> dict:
    """Return per-coordinate RMS difference over the frames both files share."""
    names_a, a = read_mot(first)
    names_b, b = read_mot(second)
    if names_a != names_b:
        raise ValueError("The two files have different coordinates")
    frames = min(len(a), len(b))
    rms = np.sqrt(np.mean((a[:frames] - b[:frames]) ** 2, axis=0))
    return {"frames_a": len(a), "frames_b": len(b), "rms": dict(zip(names_a, rms.tolist()))}


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
    print(f"frames: {result['frames_a']} vs {result['frames_b']}")
    print(f"rotations: mean RMS {np.mean(list(rotations.values())):.3f} deg, "
          f"worst {worst} {rotations[worst]:.3f} deg")
    print("translations RMS: " + ", ".join(f"{k} {rms[k] * 1000:.1f} mm" for k in sorted(TRANSLATIONS) if k in rms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
