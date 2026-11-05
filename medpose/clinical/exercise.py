"""Offline descriptive exercise review. No recommendations, dosing or progression."""
from dataclasses import dataclass

import numpy as np

from medpose.errors import ErrorTable, PROVISIONAL


@dataclass(frozen=True)
class RepetitionConfig:
    low_deg: float
    high_deg: float
    min_confidence: float
    max_gap_s: float
    status: str = PROVISIONAL

    def __post_init__(self):
        values = (self.low_deg, self.high_deg, self.min_confidence, self.max_gap_s)
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in values):
            raise ValueError("Repetition thresholds must be finite numbers")
        if self.low_deg >= self.high_deg or not 0 <= self.min_confidence <= 1 or self.max_gap_s <= 0:
            raise ValueError("Require low < high, confidence in [0,1], positive max gap")
        if self.status != PROVISIONAL:
            raise ValueError("Repetition detection is provisional: needs clinician review")


def review_exercise(trial, coordinate, config, *, plane, task, errors=None):
    """Count complete low-high-low excursions; missing data resets the detector.

    Incomplete final movements are not counted. Range and tempo are per repetition.
    Caller-selected thresholds are segmentation settings, not exercise targets.
    """
    if not isinstance(config, RepetitionConfig):
        raise ValueError("Provide RepetitionConfig")
    errors = ErrorTable.load() if errors is None else errors
    result = dict(coordinate=coordinate, repetitions=[], count=0, range_mean_deg=None,
                  duration_mean_s=None, source="provisional", threshold_status=config.status,
                  error_label=errors.lookup(trial.recording_mode, coordinate, plane, task).label,
                  excluded_frames=[], warning="Research review only; no exercise advice")
    if coordinate not in trial.coordinate_names:
        result["status"] = "coordinate unavailable"
        return result
    index = trial.coordinate_names.index(coordinate)
    if trial.units[index] != "deg":
        raise ValueError("Exercise excursion requires an angular coordinate")
    values = trial.values[:, index]
    valid = np.isfinite(values)
    if trial.confidence is not None:
        confidence = trial.confidence[:, index]
        valid &= np.isfinite(confidence) & (confidence >= config.min_confidence)
    result["confidence_status"] = "absent" if trial.confidence is None else "provided"
    result["excluded_frames"] = np.flatnonzero(~valid).tolist()
    start, high = None, None
    for i, value in enumerate(values):
        if not valid[i] or (i and trial.time[i] - trial.time[i - 1] > config.max_gap_s):
            start, high = None, None
            continue
        if value <= config.low_deg:
            if start is not None and high is not None:
                segment = values[start:i + 1]
                peak = start + int(np.argmax(segment))
                result["repetitions"].append(dict(start=start, stop=i, peak=peak,
                    range_deg=float(np.ptp(segment)), duration_s=float(trial.time[i] - trial.time[start]),
                    rising_s=float(trial.time[peak] - trial.time[start]),
                    returning_s=float(trial.time[i] - trial.time[peak])))
            start, high = i, None
        elif start is not None and value >= config.high_deg and high is None:
            high = i
    result["count"] = len(result["repetitions"])
    result["status"] = "candidate repetitions" if result["count"] else "no complete repetition"
    if result["count"]:
        result["range_mean_deg"] = float(np.mean([r["range_deg"] for r in result["repetitions"]]))
        result["duration_mean_s"] = float(np.mean([r["duration_s"] for r in result["repetitions"]]))
    return result


def side_difference(trial, left, right, config, *, plane, task, errors=None):
    a = review_exercise(trial, left, config, plane=plane, task=task, errors=errors)
    b = review_exercise(trial, right, config, plane=plane, task=task, errors=errors)
    delta = None if a["range_mean_deg"] is None or b["range_mean_deg"] is None else a["range_mean_deg"] - b["range_mean_deg"]
    return dict(left=a, right=b, left_minus_right_mean_range_deg=delta,
                error_label="error not yet measured", method="difference of mean detected repetition ranges, not paired repetitions")


def compare_exercise(before, after, coordinate, config, *, plane, task, errors=None):
    """Descriptive session difference only; no real-change label without personal MDC."""
    a = review_exercise(before, coordinate, config, plane=plane, task=task, errors=errors)
    b = review_exercise(after, coordinate, config, plane=plane, task=task, errors=errors)
    same = (before.recording_mode, sorted(before.supports), before.engine_name, before.engine_version,
            before.camera_count, before.frame_rate) == (after.recording_mode, sorted(after.supports),
            after.engine_name, after.engine_version, after.camera_count, after.frame_rate)
    delta = None if a["range_mean_deg"] is None or b["range_mean_deg"] is None else b["range_mean_deg"] - a["range_mean_deg"]
    return dict(before=a, after=b, range_difference_deg=delta if same else None,
                label="descriptive difference only; personal MDC required" if same else "not comparable",
                warnings=[] if same else ["Supports, recording mode or acquisition/engine context differ"],
                error_label="error not yet measured")
