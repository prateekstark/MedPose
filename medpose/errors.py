"""Explicit error provenance; empty by default and no guessed fallbacks.

Reliability describes a benchmark estimate versus a configured limit, not the
accuracy of an individual's body measurements. No network access is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np

from medpose.trial import MODE_CAMERAS, TRANSLATIONS, _unique_keys

CONFIG = Path(__file__).with_name("config")
SOURCES = {"measured", "simulated", "provisional"}
PROVISIONAL = "provisional: needs clinician review"


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read error configuration JSON") from exc


def _key(mode, coordinate, plane, task):
    for name, value in (("mode", mode), ("coordinate", coordinate), ("plane", plane), ("task", task)):
        _nonempty(value, name)
    if mode not in MODE_CAMERAS:
        raise ValueError("Unknown recording mode")
    return mode, coordinate, plane, task


@dataclass(frozen=True)
class ErrorEstimate:
    value: Optional[float]
    source: Optional[str]
    origin: Optional[str]
    unit: str

    def __post_init__(self):
        if self.unit not in {"deg", "m"}:
            raise ValueError("Error unit must be deg or m")
        if self.value is None:
            if self.source is not None or self.origin is not None:
                raise ValueError("Missing estimates cannot carry a source or origin")
        else:
            _number(self.value, "error value")
            if self.source not in SOURCES:
                raise ValueError("Error source must be measured, simulated or provisional")
            _nonempty(self.origin, "origin")

    @property
    def label(self):
        if self.value is None:
            return "error not yet measured"
        return f"{self.value:g} {self.unit} ({self.source}; {self.origin})"


class ErrorTable:
    """Exact (mode, coordinate, plane, task) lookup only; no wildcard/default row."""

    def __init__(self, entries=()):
        self._entries = {}
        fields = {"mode", "coordinate", "plane", "task", "value", "source", "origin", "unit"}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != fields:
                raise ValueError("Error entry has missing or unknown fields")
            key = _key(*(entry[k] for k in ("mode", "coordinate", "plane", "task")))
            if key in self._entries:
                raise ValueError("Duplicate error-table key")
            expected_unit = "m" if entry["coordinate"] in TRANSLATIONS else "deg"
            if entry["unit"] != expected_unit:
                raise ValueError("Error unit does not match coordinate units")
            if entry["value"] is None:
                raise ValueError("Omit missing error rows instead of storing a default")
            self._entries[key] = ErrorEstimate(entry["value"], entry["source"], entry["origin"], entry["unit"])

    @classmethod
    def load(cls, path=CONFIG / "errors.json"):
        data = _read(path)
        if (not isinstance(data, dict) or set(data) != {"schema_version", "entries"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1
                or not isinstance(data["entries"], list)):
            raise ValueError("Unsupported or malformed error-table schema")
        return cls(data["entries"])

    def lookup(self, mode, coordinate, plane, task):
        key = _key(mode, coordinate, plane, task)
        return self._entries.get(key, ErrorEstimate(None, None, None, "m" if coordinate in TRANSLATIONS else "deg"))


@dataclass(frozen=True)
class ReliabilityConfig:
    angular_limit_deg: float
    translation_limit_m: Optional[float]
    status: str
    origin: str

    def __post_init__(self):
        _number(self.angular_limit_deg, "angular_limit_deg")
        if self.translation_limit_m is not None:
            _number(self.translation_limit_m, "translation_limit_m")
        if self.status not in {PROVISIONAL, "cited"}:
            raise ValueError("Threshold status must be cited or provisional: needs clinician review")
        _nonempty(self.origin, "threshold origin")
        # A citation's validity/read status must be reviewed by the config author.
        if self.status == "cited" and not self.origin.startswith(("https://", "http://", "doi:")):
            raise ValueError("Cited thresholds require a citation in origin")

    @classmethod
    def load(cls, path=CONFIG / "reliability.json"):
        data = _read(path)
        expected = {"angular_limit_deg", "translation_limit_m", "status", "origin"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Malformed reliability configuration")
        return cls(**data)


@dataclass(frozen=True)
class Reliability:
    status: str
    reason: str
    error: ErrorEstimate
    limit: Optional[float]
    threshold_status: str
    threshold_origin: str


def reliability(estimate, config=None):
    """Return reliable / not reliable / unknown, retaining error and threshold provenance.

    Strictly greater than the limit is not reliable. At the limit, a measured
    estimate passes this configured check. Simulated/provisional low estimates
    never establish reliability. No translation cutoff is supplied by default.
    """
    if not isinstance(estimate, ErrorEstimate):
        raise ValueError("estimate must be an ErrorEstimate")
    config = ReliabilityConfig.load() if config is None else config
    if not isinstance(config, ReliabilityConfig):
        raise ValueError("config must be a ReliabilityConfig")
    limit = config.angular_limit_deg if estimate.unit == "deg" else config.translation_limit_m
    if estimate.value is None:
        status, reason = "unknown", "error not yet measured"
    elif limit is None:
        status, reason = "unknown", "No reliability limit configured for this unit"
    elif estimate.value > limit:
        status, reason = "not reliable", "Estimated error exceeds configured limit"
    elif estimate.source != "measured":
        status, reason = "unknown", "No measured error; simulated or provisional estimates do not establish reliability"
    else:
        status, reason = "reliable", "Measured benchmark error is within configured limit; individual error remains unknown"
    return Reliability(status, reason, estimate, limit, config.status, config.origin)
