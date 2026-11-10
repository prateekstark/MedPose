#!/usr/bin/env python3
"""Turn OpenCap output (.mot joint angles + augmented .trc markers) into a Trial.

Without this step every analysis module only ever saw synthetic data. The Trial
gets OpenCap's joint angles and, from the augmented markers ``r_calc_study`` /
``r_toe_study`` (and left equivalents), heel and toe points for gait events.

How the foot points are made (stated in the Trial's ``foot_positions.origin``):
OpenCap's world frame is y-up, and its marker augmenter shifts the whole trial so
the lowest predicted marker sits at y = 0.01 m (opencap-core utilsAugmenter.py).
A heel or toe *marker* sits on the skin or shoe, above the sole, so each marker is
moved straight down by its own low-percentile height in this trial. That makes it
an estimate of the sole point, assuming the foot reaches the floor at some time in
the trial. If the person never puts the heel down (equinus), the heel estimate will be
too low; check against video and the physio's observation.

Run this on patient output from your own terminal, never through an AI assistant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medpose.feet import FootPositions
from medpose.trial import Trial, load_opensim_mot, MODE_CAMERAS

ROOT = Path(__file__).resolve().parents[2]
VERTICAL = np.array([0.0, 1.0, 0.0])
FOOT_MARKERS = {"right": ("r_calc_study", "r_toe_study"), "left": ("L_calc_study", "L_toe_study")}

# Anatomical plane of each LaiUhlrich2022 coordinate (opencap-core opensimPipeline/Models).
OPENSIM_PLANES = {
    "pelvis_tilt": "sagittal", "pelvis_list": "frontal", "pelvis_rotation": "transverse",
    "pelvis_tx": "translation", "pelvis_ty": "translation", "pelvis_tz": "translation",
    "lumbar_extension": "sagittal", "lumbar_bending": "frontal", "lumbar_rotation": "transverse",
    **{f"{name}_{side}": plane for side in ("l", "r") for name, plane in (
        ("hip_flexion", "sagittal"), ("hip_adduction", "frontal"), ("hip_rotation", "transverse"),
        ("knee_angle", "sagittal"), ("knee_angle_beta", "coupled patella"),
        ("ankle_angle", "sagittal"), ("subtalar_angle", "subtalar (oblique axis)"),
        ("mtp_angle", "sagittal"), ("arm_flex", "sagittal"), ("arm_add", "frontal"),
        ("arm_rot", "transverse"), ("elbow_flex", "sagittal"), ("pro_sup", "transverse"))},
}
# OpenCap names the beta coordinates knee_angle_r_beta, not knee_angle_beta_r.
OPENSIM_PLANES.update({f"knee_angle_{s}_beta": "coupled patella" for s in ("l", "r")})


def planes_for(names):
    """Plane for each coordinate; unknown names are labelled 'unspecified', not guessed."""
    return {n: OPENSIM_PLANES.get(n, "unspecified") for n in names}


def read_trc(path):
    """Return (time, rate, marker names, positions (frames, markers, 3) in metres)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        header_row = next(i for i, l in enumerate(lines) if l.startswith("DataRate"))
        names_row = next(i for i, l in enumerate(lines) if l.startswith("Frame#"))
    except StopIteration:
        raise ValueError("TRC needs DataRate and Frame# header lines") from None
    keys = lines[header_row].split("\t")
    values = lines[header_row + 1].split("\t")
    meta = dict(zip([k.strip() for k in keys], [v.strip() for v in values]))
    try:
        rate = float(meta["DataRate"])
        units = meta["Units"].lower()
    except (KeyError, ValueError):
        raise ValueError("TRC header lacks DataRate or Units") from None
    scale = {"m": 1.0, "mm": 0.001, "cm": 0.01}.get(units)
    if scale is None:
        raise ValueError(f"Unsupported TRC units: {units}")
    names = [n.strip() for n in lines[names_row].split("\t")[2:] if n.strip()]
    if len(set(names)) != len(names) or not names:
        raise ValueError("TRC marker names are missing or repeated")
    rows = []
    for line in lines[names_row + 1:]:
        fields = line.split("\t")
        if not line.strip() or not fields[0].strip().lstrip("-").replace(".", "", 1).isdigit():
            continue
        fields = (fields + [""] * (2 + 3 * len(names)))[:2 + 3 * len(names)]
        rows.append([float(f) if f.strip() else np.nan for f in fields])
    if not rows:
        raise ValueError("TRC has no data rows")
    data = np.array(rows)
    positions = data[:, 2:].reshape(len(data), len(names), 3) * scale
    return data[:, 1], rate, names, positions


def foot_positions_from_markers(time, names, positions, *, floor_percentile):
    """Heel/toe sole estimates from OpenCap augmented markers (see module docstring)."""
    if not 0 <= floor_percentile <= 50:
        raise ValueError("floor_percentile must lie in [0, 50]")
    sides = {}
    for side, (heel, toe) in FOOT_MARKERS.items():
        if heel not in names or toe not in names:
            sides[side] = None
            continue
        points = positions[:, [names.index(heel), names.index(toe)], :].copy()
        for k in range(2):
            heights = points[:, k] @ VERTICAL
            finite = heights[np.isfinite(heights)]
            if not len(finite):
                points[:, k] = np.nan
                continue
            points[:, k] -= np.percentile(finite, floor_percentile) * VERTICAL
        sides[side] = points
    if sides["left"] is None and sides["right"] is None:
        return None
    origin = (f"OpenCap augmented calc/toe markers moved down by each marker's {floor_percentile:g}th-percentile "
              "height in this trial (y-up world); estimate of sole points assuming each reaches the floor")
    return FootPositions(sides["left"], sides["right"], VERTICAL, 0.0, origin)


def opencap_trial(mot_path, trc_path, *, recording_mode, camera_count, supports=(),
                  engine_version=None, floor_percentile=2.0):
    """Combine a Core .mot and its augmented .trc into one Trial."""
    trc_time, rate, names, positions = read_trc(trc_path)
    engine_version = engine_version or _core_commit()
    trial = load_opensim_mot(mot_path, recording_mode=recording_mode, engine_name="OpenCap Core",
                             engine_version=engine_version, camera_count=camera_count,
                             frame_rate=rate, supports=supports, translation_unit="m")
    # Match each angle frame to the marker frame with the same timestamp.
    index = np.searchsorted(trc_time, trial.time - 1e-6)
    index = np.clip(index, 0, len(trc_time) - 1)
    if not np.allclose(trc_time[index], trial.time, atol=min(1e-4, 0.25 / rate)):
        raise ValueError("Angle and marker files do not share timestamps; were they produced together?")
    feet = foot_positions_from_markers(trc_time[index], names, positions[index], floor_percentile=floor_percentile)
    return Trial(trial.time, trial.coordinate_names, trial.values, trial.recording_mode, trial.engine_name,
                 trial.engine_version, trial.camera_count, trial.frame_rate, trial.supports,
                 foot_positions=feet)


def _core_commit():
    try:
        lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
        return "commit " + lock["opencap-core"]["commit"][:12]
    except (OSError, KeyError, ValueError):
        return "unknown commit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mot", required=True, type=Path, help="OpenSimData/Kinematics/<trial>.mot")
    parser.add_argument("--trc", required=True, type=Path, help="MarkerData/PostAugmentation/<trial>.trc")
    parser.add_argument("--mode", required=True, choices=sorted(MODE_CAMERAS))
    parser.add_argument("--cameras", required=True, type=int)
    parser.add_argument("--support", action="append", default=[],
                        help="Equipment used, e.g. 'posterior walker', 'bilateral AFOs' (repeat). No names.")
    parser.add_argument("--floor-percentile", type=float, default=2.0,
                        help="Marker-height percentile treated as floor contact (provisional engineering choice)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    try:
        trial = opencap_trial(args.mot, args.trc, recording_mode=args.mode, camera_count=args.cameras,
                              supports=tuple(args.support), floor_percentile=args.floor_percentile)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    trial.save_json(args.output)
    feet = "with" if trial.foot_positions is not None else "WITHOUT"
    print(f"Wrote {args.output}: {len(trial.time)} frames, {len(trial.coordinate_names)} coordinates, {feet} foot points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
