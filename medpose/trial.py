"""Engine-independent, identifier-free research results; no network operations.

NaN denotes a missing observation in memory and JSON null on disk. Angles are in
 degrees; pelvis_tx/ty/tz are in metres. Free-text supports must describe equipment
 only: callers must not put identifying information in them or engine metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from medpose.feet import FootPositions

SCHEMA_VERSION = 2
TRANSLATIONS = frozenset({"pelvis_tx", "pelvis_ty", "pelvis_tz"})
MODE_CAMERAS = {"webcam": {1}, "1_phone": {1}, "2_phone": {2}, "3_4_phone": {3, 4}}


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def _array(value, field):
    try:
        result = np.array(value, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must contain numbers or missing values") from exc
    if np.isinf(result).any():
        raise ValueError(f"{field} contains infinity")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Trial:
    """Aligned observations, without subject identifiers or recording paths.

    Frame rate is acquisition metadata; time is authoritative for analysis and
    may contain gaps. No interpolation or confidence is invented. Empty trials
    are allowed, with values/confidence shaped (0, number_of_coordinates).
    """
    time: np.ndarray
    coordinate_names: Tuple[str, ...]
    values: np.ndarray
    recording_mode: str
    engine_name: str
    engine_version: str
    camera_count: int
    frame_rate: float
    supports: Tuple[str, ...] = ()
    confidence: Optional[np.ndarray] = None
    foot_positions: Optional[FootPositions] = None

    def __post_init__(self):
        if isinstance(self.coordinate_names, str) or not self.coordinate_names:
            raise ValueError("coordinate_names must be a nonempty sequence")
        names = tuple(_text(n, "coordinate name") for n in self.coordinate_names)
        if len(set(names)) != len(names) or "time" in names:
            raise ValueError("Coordinate names must be unique and cannot be time")
        time = _array(self.time, "time")
        values = _array(self.values, "values")
        if time.ndim != 1 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("time must be finite, one-dimensional and strictly increasing")
        if values.shape != (len(time), len(names)):
            raise ValueError("values must have shape (frames, coordinates)")
        if not isinstance(self.recording_mode, str) or self.recording_mode not in MODE_CAMERAS:
            raise ValueError("Unknown recording mode")
        if (isinstance(self.camera_count, (bool, np.bool_))
                or not isinstance(self.camera_count, (int, np.integer))
                or self.camera_count not in MODE_CAMERAS[self.recording_mode]):
            raise ValueError("camera_count is inconsistent with recording_mode")
        if (isinstance(self.frame_rate, (bool, np.bool_))
                or not isinstance(self.frame_rate, (int, float, np.number))
                or not np.isfinite(self.frame_rate) or self.frame_rate <= 0):
            raise ValueError("frame_rate must be a positive finite number")
        _text(self.engine_name, "engine_name")
        _text(self.engine_version, "engine_version")
        if isinstance(self.supports, str):
            raise ValueError("supports must be a sequence of equipment descriptions")
        supports = tuple(_text(s, "support") for s in self.supports)
        confidence = None
        if self.confidence is not None:
            confidence = _array(self.confidence, "confidence")
            if confidence.shape != values.shape:
                raise ValueError("confidence must match values shape")
            known = confidence[np.isfinite(confidence)]
            if np.any((known < 0) | (known > 1)):
                raise ValueError("confidence must be in [0, 1] or missing")
        if self.foot_positions is not None:
            if not isinstance(self.foot_positions, FootPositions):
                raise ValueError("foot_positions must be FootPositions")
            self.foot_positions.validate_frames(len(time))
        for field, value in (("time", time), ("values", values), ("coordinate_names", names),
                             ("supports", supports), ("confidence", confidence)):
            object.__setattr__(self, field, value)

    @property
    def units(self):
        return tuple("m" if name in TRANSLATIONS else "deg" for name in self.coordinate_names)

    def coordinate(self, name):
        try:
            return self.values[:, self.coordinate_names.index(name)]
        except ValueError:
            raise ValueError(f"Coordinate unavailable: {name}") from None

    def to_dict(self):
        def rows(array):
            return [[None if np.isnan(v) else float(v) for v in row] for row in array]
        return {"schema_version": SCHEMA_VERSION, "time": self.time.tolist(),
                "coordinate_names": list(self.coordinate_names), "units": list(self.units),
                "values": rows(self.values), "recording_mode": self.recording_mode,
                "engine_name": self.engine_name, "engine_version": self.engine_version,
                "camera_count": int(self.camera_count), "frame_rate": float(self.frame_rate),
                "supports": list(self.supports),
                "confidence": None if self.confidence is None else rows(self.confidence),
                "foot_positions": None if self.foot_positions is None else self.foot_positions.to_dict()}

    @classmethod
    def from_dict(cls, data):
        fields = {"schema_version", "time", "coordinate_names", "units", "values",
                  "recording_mode", "engine_name", "engine_version", "camera_count",
                  "frame_rate", "supports", "confidence"}
        if isinstance(data, dict) and data.get("schema_version") == 2:
            fields.add("foot_positions")
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("Trial JSON has missing or unknown fields (identifiers are not supported)")
        if type(data["schema_version"]) is not int or data["schema_version"] not in (1, SCHEMA_VERSION):
            raise ValueError("Unsupported Trial schema version")
        for field in ("time", "coordinate_names", "units", "values", "supports"):
            if not isinstance(data[field], list):
                raise ValueError(f"{field} must be a JSON array")
        for field in ("time", "values", "confidence"):
            value = data[field]
            if field == "confidence" and value is None:
                continue
            if not isinstance(value, list):
                raise ValueError(f"{field} must be a JSON array")
            flat = value if field == "time" else [v for row in value if isinstance(row, list) for v in row]
            if field != "time" and any(not isinstance(row, list) for row in value):
                raise ValueError(f"{field} must contain arrays")
            if any(v is not None and (type(v) not in (int, float) or not np.isfinite(v)) for v in flat):
                raise ValueError(f"{field} must contain finite JSON numbers or null")
        kwargs = {key: data[key] for key in fields - {"schema_version", "units"}}
        if data["time"] == []:
            for field in ("values", "confidence"):
                if data[field] == []:
                    kwargs[field] = np.empty((0, len(data["coordinate_names"])))
        if kwargs.get("foot_positions") is not None:
            kwargs["foot_positions"] = FootPositions.from_dict(kwargs["foot_positions"], len(data["time"]))
        trial = cls(**kwargs)
        if list(trial.units) != data["units"]:
            raise ValueError("Coordinate units must be degrees or metres for pelvis translations")
        return trial

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), allow_nan=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read Trial JSON") from exc
        return cls.from_dict(data)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_opensim_mot(path, *, recording_mode, engine_name, engine_version,
                     camera_count, frame_rate, supports=(), confidence=None,
                     angle_unit=None, translation_unit=None):
    """Read a .mot file; inspired by tools/compare_mot.py's header parser.

    inDegrees=yes/no controls angular conversion. Without it, the caller must
    supply angle_unit='deg' or 'rad'. Conflicting declarations are refused.
    If pelvis translations exist, the caller must confirm translation_unit='m'.
    Header names/paths are discarded, never copied into Trial metadata.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("Could not read OpenSim MOT file") from exc
    try:
        end = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration:
        raise ValueError("MOT file has no endheader") from None
    metadata = {}
    for line in lines[:end]:
        parts = line.strip().replace("=", " ", 1).split()
        if len(parts) == 2 and parts[0].lower() in {"indegrees", "nrows", "ncolumns", "datarows", "datacolumns"}:
            key = parts[0].lower()
            if key in metadata:
                raise ValueError(f"Duplicate MOT header field: {key}")
            metadata[key] = parts[1]
    body = [line.split() for line in lines[end + 1:] if line.strip()]
    if not body or len(body[0]) < 2 or body[0][0] != "time":
        raise ValueError("MOT must have time first and at least one coordinate")
    names, rows = body[0][1:], body[1:]
    if not rows or any(len(row) != len(names) + 1 for row in rows):
        raise ValueError("MOT has no observations or inconsistent row widths")
    try:
        array = np.asarray(rows, dtype=float)
        for key, expected in (("nrows", len(rows)), ("datarows", len(rows)),
                              ("ncolumns", len(names) + 1), ("datacolumns", len(names) + 1)):
            if key in metadata and int(metadata[key]) != expected:
                raise ValueError("MOT header dimensions do not match observations")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid numeric MOT data or header dimensions") from exc
    declared = metadata.get("indegrees")
    if declared is not None:
        if declared.lower() not in {"yes", "no"}:
            raise ValueError("MOT inDegrees must be yes or no")
        declared = "deg" if declared.lower() == "yes" else "rad"
        if angle_unit is not None and declared != angle_unit:
            raise ValueError("Conflicting angular unit declarations")
    unit = declared or angle_unit
    if unit not in {"deg", "rad"}:
        raise ValueError("Angular units absent; supply angle_unit='deg' or 'rad'")
    if any(n in TRANSLATIONS for n in names) and translation_unit != "m":
        raise ValueError("Confirm pelvis translations are metres with translation_unit='m'")
    values = array[:, 1:].copy()
    if unit == "rad":
        for i, name in enumerate(names):
            if name not in TRANSLATIONS:
                values[:, i] = np.rad2deg(values[:, i])
    return Trial(array[:, 0], tuple(names), values, recording_mode, engine_name,
                 engine_version, camera_count, frame_rate, tuple(supports), confidence)
