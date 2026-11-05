"""Optional world-frame foot contact-point observations (metres).

Points are heel and toe *sole contact locations*, not uncorrected skin markers.
Their mapping to sole locations must be supplied by an engine, never assumed.
Ground plane: dot(unit_normal, position) = offset_m; normal points upward.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FootPositions:
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    normal: np.ndarray
    offset_m: float
    origin: str
    left_confidence: Optional[np.ndarray] = None
    right_confidence: Optional[np.ndarray] = None

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1):
            raise ValueError("Ground normal must be a finite unit vector")
        if isinstance(self.offset_m, bool) or not isinstance(self.offset_m, (int, float)) or not np.isfinite(self.offset_m):
            raise ValueError("Ground offset must be finite metres")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ValueError("Foot origin must describe how sole locations were obtained; no identifiers")
        normal = normal.copy()
        normal.setflags(write=False)
        object.__setattr__(self, "normal", normal)
        for side in ("left", "right"):
            points, confidence = getattr(self, side), getattr(self, side + "_confidence")
            if points is None:
                if confidence is not None:
                    raise ValueError("Foot confidence requires positions")
                continue
            points = np.array(points, dtype=float, copy=True)
            if points.ndim != 3 or points.shape[1:] != (2, 3) or np.isinf(points).any():
                raise ValueError("Foot positions must have shape (frames, heel/toe, xyz), in metres")
            points.setflags(write=False)
            object.__setattr__(self, side, points)
            if confidence is not None:
                confidence = np.array(confidence, dtype=float, copy=True)
                if confidence.shape != points.shape[:2] or np.isinf(confidence).any():
                    raise ValueError("Foot confidence must have shape (frames, heel/toe)")
                if np.any((confidence < 0) | (confidence > 1)):
                    raise ValueError("Foot confidence must lie in [0,1] or be missing")
                confidence.setflags(write=False)
                object.__setattr__(self, side + "_confidence", confidence)

    def validate_frames(self, frames):
        for side in ("left", "right"):
            points = getattr(self, side)
            if points is not None and len(points) != frames:
                raise ValueError("Foot observations must align with Trial frames")

    def to_dict(self):
        def clean(value):
            if isinstance(value, list):
                return [clean(v) for v in value]
            return None if np.isnan(value) else value
        return {"unit": "m", "frame": "world", "point_order": ["heel_sole", "toe_sole"],
                "normal": self.normal.tolist(), "offset_m": self.offset_m, "origin": self.origin,
                **{key: None if getattr(self, key) is None else clean(getattr(self, key).tolist())
                   for key in ("left", "right", "left_confidence", "right_confidence")}}

    @classmethod
    def from_dict(cls, data, frames):
        fields = {"unit", "frame", "point_order", "normal", "offset_m", "origin",
                  "left", "right", "left_confidence", "right_confidence"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("Malformed foot positions schema")
        if data["unit"] != "m" or data["frame"] != "world" or data["point_order"] != ["heel_sole", "toe_sole"]:
            raise ValueError("Foot positions require world metres and heel/toe sole point order")
        def validate_numbers(value):
            if isinstance(value, list):
                for child in value:
                    validate_numbers(child)
            elif value is not None and (type(value) not in (float, int) or not np.isfinite(value)):
                raise ValueError("Foot arrays require finite JSON numbers or null")
        kwargs = {key: data[key] for key in fields - {"unit", "frame", "point_order"}}
        for key in ("normal", "left", "right", "left_confidence", "right_confidence"):
            validate_numbers(kwargs[key])
            if frames == 0 and kwargs[key] == []:
                kwargs[key] = np.empty((0, 2) if key.endswith("confidence") else (0, 2, 3))
        feet = cls(**kwargs)
        feet.validate_frames(frames)
        return feet
