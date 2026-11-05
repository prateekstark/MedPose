"""Provisional geometric contact candidates, not force/load measurements.

No cutoff defaults are supplied: each geometric/time/confidence threshold must be
chosen explicitly and is provisional pending clinician review. A low sole point
is evidence of proximity, not proof of weight bearing, especially with supports.
"""
from dataclasses import dataclass, asdict

import numpy as np

from medpose.errors import ErrorTable


@dataclass(frozen=True)
class ContactConfig:
    clearance_m: float
    penetration_tolerance_m: float
    min_confidence: float
    max_gap_s: float
    min_stance_s: float
    min_swing_s: float
    status: str = "provisional: needs clinician review"

    def __post_init__(self):
        for name in ("clearance_m", "penetration_tolerance_m", "min_confidence",
                     "max_gap_s", "min_stance_s", "min_swing_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a nonnegative finite number")
        if self.min_confidence > 1 or min(self.max_gap_s, self.min_stance_s, self.min_swing_s) <= 0:
            raise ValueError("Confidence must be <=1; time thresholds must be positive")
        if self.status != "provisional: needs clinician review":
            raise ValueError("This detector is provisional; no clinical validation is established")


def detect_contacts(trial, side, config):
    """Detect first sampled near-ground frames and departures, without bridging gaps.

    A complete cycle requires two observed initial contacts and exactly one
    departure between them. Contact at the first frame is not an observed onset.
    Returned events are candidate kinematic contacts, never verified contacts.
    """
    if side not in ("left", "right") or not isinstance(config, ContactConfig):
        raise ValueError("Expected left/right and ContactConfig")
    result = dict(status="detection failed", reason="foot positions unavailable", initial_contacts=[],
                  departures=[], cycles=[], low_confidence_frames=[], unknown_confidence_frames=[],
                  invalid_frames=[], config=asdict(config), source="provisional",
                  error_label="error not yet measured")
    feet = trial.foot_positions
    if feet is None or getattr(feet, side) is None:
        return result
    points = getattr(feet, side)
    if len(trial.time) < 3:
        result["reason"] = "not enough frames"
        return result
    heights = points @ feet.normal - feet.offset_m
    valid = np.isfinite(points).all(axis=(1, 2)) & (heights >= -config.penetration_tolerance_m).all(axis=1)
    confidence = getattr(feet, side + "_confidence")
    if confidence is None:
        result["unknown_confidence_frames"] = list(range(len(points)))
    else:
        unknown = ~np.isfinite(confidence).all(axis=1)
        low = (confidence < config.min_confidence).any(axis=1)
        result["low_confidence_frames"] = np.flatnonzero(low).tolist()
        result["unknown_confidence_frames"] = np.flatnonzero(unknown).tolist()
        valid &= ~unknown & ~low
    result["invalid_frames"] = np.flatnonzero(~valid).tolist()
    contact = (heights <= config.clearance_m).any(axis=1)
    connected = valid[1:] & valid[:-1] & (np.diff(trial.time) <= config.max_gap_s)
    starts = np.flatnonzero(connected & contact[1:] & ~contact[:-1]) + 1
    ends = np.flatnonzero(connected & ~contact[1:] & contact[:-1]) + 1
    result["initial_contacts"], result["departures"] = starts.tolist(), ends.tolist()
    for start, stop in zip(starts[:-1], starts[1:]):
        departures = ends[(ends > start) & (ends < stop)]
        if len(departures) != 1 or not valid[start:stop + 1].all() or not connected[start:stop].all():
            continue
        departure = int(departures[0])
        stance = trial.time[departure] - trial.time[start]
        swing = trial.time[stop] - trial.time[departure]
        if stance < config.min_stance_s or swing < config.min_swing_s:
            continue
        result["cycles"].append(dict(start=int(start), departure=departure, stop=int(stop)))
    if result["cycles"]:
        result["status"], result["reason"] = "candidate cycles", "provisional geometric proximity; load is unobserved"
    else:
        result["reason"] = "no complete valid cycle; no events inferred across missing data"
    return result


def _summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(per_cycle=[float(v) if np.isfinite(v) else None for v in values], mean=float(finite.mean()) if len(finite) else None,
                # Sample SD needs two observations. Zero is not substituted for one cycle.
                sd=float(finite.std(ddof=1)) if len(finite) > 1 else None, count=int(len(finite)))


def analyse_gait(trial, config, *, planes, task="walking", errors=None, samples=101, progression_direction=None):
    """Normalize complete cycles without interpolating over missing/low-confidence values.

    Cadence is 120 / ipsilateral cycle duration (an alternating-step assumption),
    labelled provisional. Stance is geometric proximity duration / cycle duration.
    Speed/step length are unavailable here: no progression axis or foot-to-pelvis
    frame relationship is guessed. No forces are calculated.
    """
    if type(samples) is not int or samples < 2:
        raise ValueError("samples must be an integer >=2")
    if not isinstance(planes, dict) or any(n not in planes or not isinstance(planes[n], str) or not planes[n] for n in trial.coordinate_names):
        raise ValueError("Provide an explicit anatomical plane for every coordinate")
    errors = ErrorTable.load() if errors is None else errors
    output = dict(source="provisional", error_label="error not yet measured", sides={}, differences={},
                  speed=dict(status="unavailable", reason="explicit progression-axis analysis not implemented", error_label="error not yet measured"),
                  step_length=dict(status="unavailable", reason="explicit progression-axis analysis not implemented", error_label="error not yet measured"))
    for side in ("left", "right"):
        events = detect_contacts(trial, side, config)
        cycles = events["cycles"]
        durations = [trial.time[c["stop"]] - trial.time[c["start"]] for c in cycles]
        stance = [100 * (trial.time[c["departure"]] - trial.time[c["start"]]) / duration for c, duration in zip(cycles, durations)]
        data = dict(events=events, duration_s=_summary(durations), stance_percent=_summary(stance),
                    cadence_steps_min=_summary([120 / d for d in durations]), coordinates={})
        for j, name in enumerate(trial.coordinate_names):
            normalized, ranges, peaks, troughs, low_frames = [], [], [], [], []
            if trial.confidence is not None:
                low_frames = np.flatnonzero(np.isnan(trial.confidence[:, j]) | (trial.confidence[:, j] < config.min_confidence)).tolist()
            for cycle in cycles:
                start, stop = cycle["start"], cycle["stop"]
                values = trial.values[start:stop + 1, j]
                valid = np.isfinite(values).all() and not any(start <= f <= stop for f in low_frames)
                if not valid:
                    normalized.append(None)
                    ranges.append(np.nan); peaks.append(np.nan); troughs.append(np.nan)
                    continue
                phase = (trial.time[start:stop + 1] - trial.time[start]) / (trial.time[stop] - trial.time[start])
                normalized.append(np.interp(np.linspace(0, 1, samples), phase, values).tolist())
                ranges.append(float(np.ptp(values))); peaks.append(float(values.max())); troughs.append(float(values.min()))
            estimate = errors.lookup(trial.recording_mode, name, planes[name], task)
            data["coordinates"][name] = dict(normalized=normalized, range=_summary(ranges),
                peak=_summary(peaks), trough=_summary(troughs), error_label=estimate.label,
                low_or_missing_confidence_frames=low_frames,
                confidence_status="absent" if trial.confidence is None else "provided")
        output["sides"][side] = data
    # Difference of side-specific mean ranges, not unpaired frame subtraction.
    for name in trial.coordinate_names:
        if name.endswith("_l") and name[:-2] + "_r" in trial.coordinate_names:
            other = name[:-2] + "_r"
            left = output["sides"]["left"]["coordinates"][name]["range"]["mean"]
            right = output["sides"]["right"]["coordinates"][other]["range"]["mean"]
            output["differences"][name[:-2]] = dict(left_minus_right_range=None if left is None or right is None else left - right,
                error_label="error not yet measured", method="difference of side-specific mean cycle ranges")
    if progression_direction is not None:
        axis = np.asarray(progression_direction, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
            raise ValueError("progression_direction must be a finite unit vector in the shared world frame")
        if trial.foot_positions is not None and not np.isclose(axis @ trial.foot_positions.normal, 0):
            raise ValueError("Progression direction must lie in the ground plane")
        names = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
        if all(n in trial.coordinate_names for n in names):
            indices = [trial.coordinate_names.index(n) for n in names]
            pelvis = trial.values[:, indices]
            provenance = [errors.lookup(trial.recording_mode, n, planes[n], task).label for n in names]
            speed = {}
            for side in ("left", "right"):
                values = []
                for c in output["sides"][side]["events"]["cycles"]:
                    a, b = c["start"], c["stop"]
                    valid = np.isfinite(pelvis[a:b + 1]).all()
                    if trial.confidence is not None:
                        conf = trial.confidence[a:b + 1, indices]
                        valid = valid and np.isfinite(conf).all() and (conf >= config.min_confidence).all()
                    values.append(float((pelvis[b] - pelvis[a]) @ axis / (trial.time[b] - trial.time[a])) if valid else np.nan)
                speed[side] = _summary(values)
            output["speed"] = dict(status="signed progression speed", sides=speed, unit="m/s",
                pelvis_error_labels=provenance, error_label="error not yet measured",
                caveat="Pelvis error provenance is retained; no propagated speed error has been measured")
            steps = {"left": [], "right": []}
            feet = trial.foot_positions
            if feet is not None and feet.left is not None and feet.right is not None:
                for side, sign in (("left", 1), ("right", -1)):
                    for frame in output["sides"][side]["events"]["initial_contacts"]:
                        points = np.concatenate([feet.left[frame], feet.right[frame]])
                        valid = np.isfinite(points).all() and np.isfinite(pelvis[frame]).all()
                        for check_side in ("left", "right"):
                            event = output["sides"][check_side]["events"]
                            valid = valid and frame not in event["invalid_frames"]
                        if valid:
                            # Step length: leading minus trailing heel-sole projection at onset.
                            steps[side].append(float(sign * (feet.left[frame, 0] - feet.right[frame, 0]) @ axis))
                output["step_length"] = dict(status="candidate contact heel-to-heel projection", unit="m",
                    sides={side: _summary(values) for side, values in steps.items()},
                    pelvis_error_labels=provenance, error_label="error not yet measured",
                    caveat="Foot-location and event errors are unmeasured; pelvis labels do not quantify step-length error")
        else:
            for metric in ("speed", "step_length"):
                output[metric]["reason"] = "pelvis translations unavailable"
    return output
