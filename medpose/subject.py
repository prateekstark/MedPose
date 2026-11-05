"""Measured subject data, validated before any pipeline runs.

Body size drives model scaling, so a wrong value produces plausible-looking but wrong joint
angles. OpenCap Monocular silently substitutes an adult default (1.70 m, 70 kg, male) for
missing fields, so MedPose requires measured values and has no defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Wide on purpose: these catch unit mistakes (cm for m, lb for kg) and blanks, not unusual bodies.
HEIGHT_RANGE_M = (0.5, 2.5)
MASS_RANGE_KG = (5.0, 300.0)
SEXES = {"male", "female"}
# The example values in OPENCAP_START.md, and the adult default upstream falls back to.
TEMPLATE_VALUES = (1.70, 70.0)


class SubjectError(ValueError):
    """Subject data is missing or can't be right."""


@dataclass(frozen=True)
class Subject:
    height_m: float
    mass_kg: float
    sex: Optional[str] = None

    @property
    def looks_like_template(self) -> bool:
        return (self.height_m, self.mass_kg) == TEMPLATE_VALUES


def _number(data: dict, key: str, limits: tuple, source: str) -> float:
    value = data.get(key)
    if value is None:
        raise SubjectError(f"{source} is missing {key}; enter the measured value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubjectError(f"{source}: {key} must be a number, got {value!r}")
    if not limits[0] <= value <= limits[1]:
        raise SubjectError(f"{source}: {key}={value} is outside {limits[0]}-{limits[1]}; check the units")
    return float(value)


def _read_yaml(path: Path) -> dict:
    try:
        with Path(path).open() as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise SubjectError(f"Could not read subject data from {path}: {e}") from e
    if not isinstance(data, dict):
        raise SubjectError(f"{path} doesn't contain subject fields")
    return data


def _warn_if_template(subject: Subject, source: Path) -> None:
    if subject.looks_like_template:
        print(f"Warning: {source} has the template values {TEMPLATE_VALUES[0]} m and "
              f"{TEMPLATE_VALUES[1]} kg. Confirm these were really measured.")


def load_mono_subject(path: Path) -> Subject:
    """Read a Monocular subject YAML, which needs height_m, mass_kg and sex."""
    data = _read_yaml(path)
    height = _number(data, "height_m", HEIGHT_RANGE_M, str(path))
    mass = _number(data, "mass_kg", MASS_RANGE_KG, str(path))
    sex = data.get("sex")
    if sex not in SEXES:
        raise SubjectError(f"{path}: sex must be one of {sorted(SEXES)}, as the model expects; got {sex!r}")
    subject = Subject(height, mass, sex)
    _warn_if_template(subject, path)
    return subject


def load_core_session(path: Path) -> Subject:
    """Read a Core sessionMetadata.yaml. Core uses height and mass; sex is optional there."""
    data = _read_yaml(path)
    subject = Subject(_number(data, "height_m", HEIGHT_RANGE_M, str(path)),
                      _number(data, "mass_kg", MASS_RANGE_KG, str(path)),
                      data.get("gender_mf"))
    _warn_if_template(subject, path)
    return subject
