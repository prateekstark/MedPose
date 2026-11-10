"""Load a reference recording (gait-lab C3D, CSV export or OpenSim .mot) as a Trial.

Hospital gait labs usually export C3D files holding markers plus their own model's
joint angles (for example Plug-in Gait's ``LKneeAngles``), and gait events. MedPose
predicts OpenSim coordinates, so a reference is only comparable through an explicit
mapping file (``benchmarks/conventions/*.json``). Unmapped or disabled angles are
dropped, never guessed.

C3D reading uses the pure-Python ``c3d`` package (MIT), imported only when needed.
Nothing here is uploaded; patient files must stay under ``patient_data/``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from medpose.trial import Trial, TRANSLATIONS, _unique_keys

CONVENTIONS = Path(__file__).with_name("conventions")
AXES = {"X": 0, "Y": 1, "Z": 2}
STATUSES = {"enabled", "caution", "disabled"}


def _strip_subject(label):
    """Vicon prefixes model outputs with the subject name ("Name:LKneeAngles")."""
    return label.strip().split(":")[-1]


def read_c3d(path):
    """Return dict(time, rate, points={label: (frames, 3)}, units, events=[...]).

    Invalid samples (negative residual) become NaN. Times follow the C3D convention
    that frame 1 is t = 0, so cropped files keep their original clock and events
    line up. Event entries are dict(context, label, time_s).
    """
    try:
        import c3d  # noqa: WPS433 (optional dependency)
    except ImportError as exc:
        raise RuntimeError("Reading C3D needs the 'c3d' package: pip install c3d==0.5.2") from exc
    import warnings
    with open(path, "rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "No analog data" is normal for exported angles.
        reader = c3d.Reader(handle)
        labels = [_strip_subject(l) for l in reader.point_labels]
        rate = float(reader.point_rate)
        frames, points = [], []
        for number, frame_points, _ in reader.read_frames():
            frames.append(number)
            xyz = np.array(frame_points[:, :3], dtype=float)
            xyz[frame_points[:, 3] < 0] = np.nan
            points.append(xyz)
        events = []
        times = reader.get("EVENT:TIMES")
        if times is not None:
            # Stored as (minutes, seconds) pairs in file order, whatever the declared shape.
            pairs = np.asarray(times.float_array, dtype=float).ravel().reshape(-1, 2)
            def strings(key):
                param = reader.get(key)
                return [] if param is None else [s.strip() for s in np.atleast_1d(param.string_array)]
            event_labels, contexts = strings("EVENT:LABELS"), strings("EVENT:CONTEXTS")
            for i, (minutes, seconds) in enumerate(pairs):
                events.append(dict(context=contexts[i] if i < len(contexts) else "",
                                   label=event_labels[i] if i < len(event_labels) else "",
                                   time_s=float(60 * minutes + seconds)))
        angle_units = reader.get("POINT:ANGLE_UNITS")
        units = dict(angle=angle_units.string_value.strip() if angle_units is not None else None,
                     position=reader.get("POINT:UNITS").string_value.strip() if reader.get("POINT:UNITS") else None)
    if len(set(labels)) != len(labels):
        raise ValueError("C3D point labels repeat after removing subject prefixes")
    array = np.array(points)  # (frames, labels, 3)
    time = (np.asarray(frames, dtype=float) - 1) / rate
    return dict(time=time, rate=rate, points={l: array[:, i] for i, l in enumerate(labels)},
                units=units, events=sorted(events, key=lambda e: e["time_s"]))


def c3d_channels(c3d_data):
    """Flatten C3D points into "Label.X/Y/Z" channels for mapping."""
    return {f"{label}.{axis}": values[:, i] for label, values in c3d_data["points"].items()
            for axis, i in AXES.items()}


def read_table(path, *, time_column="time"):
    """Read a CSV export with a header row and a time column in seconds."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or time_column not in rows[0]:
        raise ValueError(f"CSV needs a header with a '{time_column}' column")
    header = [h.strip() for h in rows[0]]
    if len(set(header)) != len(header):
        raise ValueError("CSV header repeats a column")
    body = [r for r in rows[1:] if any(c.strip() for c in r)]
    if not body or any(len(r) != len(header) for r in body):
        raise ValueError("CSV has no rows or inconsistent row widths")
    try:
        data = np.array([[float(c) if c.strip() not in ("", "nan", "NaN") else np.nan for c in r] for r in body])
    except ValueError as exc:
        raise ValueError("CSV values must be numeric or empty") from exc
    columns = {h: data[:, i] for i, h in enumerate(header)}
    return columns.pop(time_column), columns


def load_mapping(name_or_path="plugin_gait"):
    path = Path(name_or_path)
    if not path.suffix:
        path = CONVENTIONS / f"{name_or_path}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read convention mapping {path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("entries"), list):
        raise ValueError("Unsupported convention mapping schema")
    seen_ref, seen_coord = set(), set()
    for entry in data["entries"]:
        if set(entry) != {"reference", "coordinate", "sign", "plane", "status", "note"}:
            raise ValueError("Mapping entry has missing or unknown fields")
        if entry["status"] not in STATUSES:
            raise ValueError("Mapping status must be enabled, caution or disabled")
        if entry["status"] != "disabled" and entry["sign"] not in (1, -1):
            raise ValueError("Enabled mappings need sign 1 or -1")
        if entry["reference"] in seen_ref or entry["coordinate"] in seen_coord:
            raise ValueError("Mapping repeats a reference channel or coordinate")
        seen_ref.add(entry["reference"]); seen_coord.add(entry["coordinate"])
    return data


def reference_trial(time, channels, mapping, *, recording_mode, camera_count, frame_rate,
                    reference_name, include_caution=True):
    """Build a Trial named in OpenSim coordinates from mapped reference channels.

    Returns (trial, report) where report lists used, caution, disabled and absent
    mappings. ``recording_mode``/``camera_count``/``frame_rate`` are copied from the
    prediction so benchmark code can pair the two; ``reference_name`` records what the
    reference really was (e.g. "Vicon Plug-in Gait, hospital lab").
    """
    if isinstance(mapping, (str, Path)):
        mapping = load_mapping(mapping)
    names, columns, report = [], [], dict(used=[], caution=[], disabled=[], absent=[])
    for entry in mapping["entries"]:
        if entry["status"] == "disabled":
            report["disabled"].append(entry["reference"])
            continue
        if entry["status"] == "caution" and not include_caution:
            report["disabled"].append(entry["reference"])
            continue
        if entry["reference"] not in channels:
            report["absent"].append(entry["reference"])
            continue
        if entry["coordinate"] in TRANSLATIONS:
            raise ValueError("Translation mappings need explicit unit handling; not supported")
        names.append(entry["coordinate"])
        columns.append(entry["sign"] * np.asarray(channels[entry["reference"]], dtype=float))
        report["caution" if entry["status"] == "caution" else "used"].append(entry["coordinate"])
    if not names:
        raise ValueError("No mapped reference channels were found; check labels and mapping")
    trial = Trial(np.asarray(time, dtype=float), tuple(names), np.column_stack(columns), recording_mode,
                  reference_name, mapping.get("name", "mapping"), camera_count, frame_rate)
    return trial, report


def reference_events(events):
    """Group C3D events into {"left"/"right": {"foot_strike": [...], "foot_off": [...]}} seconds."""
    grouped = {side: {"foot_strike": [], "foot_off": []} for side in ("left", "right")}
    for event in events:
        side = event["context"].strip().lower()
        label = event["label"].strip().lower().replace(" ", "_")
        if side in grouped and label in grouped[side]:
            grouped[side][label].append(event["time_s"])
    for side in grouped.values():
        for key in side:
            side[key].sort()
    return grouped
