"""Repeatability calculations, not evidence of treatment response.

ICC(A,1): Liljequist et al. (2019), Eq.12, two-way absolute agreement,
single measurement, https://doi.org/10.1371/journal.pone.0219854 .
MDC95/repeatability: 1.96*sqrt(2)*within-person SD, Martin Bland,
https://www-users.york.ac.uk/~mb55/meas/accept.htm .
Requires stable, independent repeat sessions with comparable measurement context.
"""
from dataclasses import dataclass

import numpy as np

from medpose.errors import SOURCES
from medpose.trial import Trial

ICC_ORIGIN = "https://doi.org/10.1371/journal.pone.0219854 (Eq.12)"
MDC_ORIGIN = "https://www-users.york.ac.uk/~mb55/meas/accept.htm"


def icc_absolute_agreement(values, *, source):
    """Balanced subjects x repeat sessions. Complete-case subjects only, counted.

    This is a cohort statistic, not a within-person ICC. Missing entire rows are
    excluded rather than imputing sessions. Negative ICCs are not clamped to zero.
    No ICC quality bands or confidence interval are supplied.
    """
    if source not in SOURCES:
        raise ValueError("Specify measured, simulated or provisional source")
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or np.isinf(data).any():
        raise ValueError("ICC requires a subjects x sessions matrix without infinities")
    usable = np.isfinite(data).all(axis=1)
    excluded = int((~usable).sum())
    data = data[usable]
    n, k = data.shape
    result = dict(value=None, source=source, origin=ICC_ORIGIN,
                  excluded_subjects=excluded, status="not enough sessions or subjects")
    if n < 2 or k < 2:
        return result
    grand = data.mean()
    rows, cols = data.mean(axis=1), data.mean(axis=0)
    ms_subject = k * np.sum((rows - grand) ** 2) / (n - 1)
    ms_session = n * np.sum((cols - grand) ** 2) / (k - 1)
    residual = data - rows[:, None] - cols[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = ms_subject + (k - 1) * ms_error + k / n * (ms_session - ms_error)
    result["status"] = "undefined: zero denominator" if denominator == 0 else "computed"
    if denominator != 0:
        result["value"] = float((ms_subject - ms_error) / denominator)
    return result


@dataclass(frozen=True)
class SessionMeasure:
    """A named scalar summary derived from a Trial, never an identifier.

    The caller must use the same metric definition, units and task for every
    session. NaN is missing. No anonymous API can verify the person's identity;
    the caller must supply only this person's own stable baseline sessions.
    """
    trial: Trial
    metric: str
    unit: str
    value: float

    def __post_init__(self):
        if not isinstance(self.trial, Trial) or not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("SessionMeasure requires Trial and a named metric")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("Metric unit is required")
        if isinstance(self.value, bool) or not isinstance(self.value, (float, int)) or np.isinf(self.value):
            raise ValueError("Session value must be finite or NaN")


def _context(measure):
    trial = measure.trial
    return (measure.metric, measure.unit, trial.recording_mode, tuple(sorted(trial.supports)),
            trial.engine_name, trial.engine_version, trial.camera_count, trial.frame_rate)


def compare_change(before, after, own_repeat_sessions, *, source, min_repeat_sessions,
                   stable_baseline_confirmed):
    """Single-session change versus personal baseline MDC95.

    min_repeat_sessions is an explicit provisional study-design choice, >=2;
    two is only the mathematical minimum for sample SD, not clinical adequacy.
    Baseline sessions must be independent of the compared before/after sessions.
    Stable-baseline confirmation is required; trend is not silently called noise.
    Only measured-source MDC can produce 'exceeds measured noise'.
    """
    if source not in SOURCES:
        raise ValueError("Specify measured, simulated or provisional source")
    if type(min_repeat_sessions) is not int or min_repeat_sessions < 2:
        raise ValueError("min_repeat_sessions must be an integer >=2")
    if type(stable_baseline_confirmed) is not bool:
        raise ValueError("stable_baseline_confirmed must be explicit bool")
    repeats = list(own_repeat_sessions)
    if any(not isinstance(m, SessionMeasure) for m in [before, after] + repeats):
        raise ValueError("All inputs must be SessionMeasure")
    if len({id(m.trial) for m in repeats}) != len(repeats):
        raise ValueError("Repeat sessions must be distinct Trials")
    if any(m.trial is before.trial or m.trial is after.trial for m in repeats):
        raise ValueError("Noise calibration sessions must be independent of the comparison")
    result = dict(change=None, mdc95=None, source=source, origin=MDC_ORIGIN,
                  label="not enough sessions", warnings=[], excluded_sessions=0,
                  min_repeat_sessions=min_repeat_sessions,
                  threshold_status="provisional: needs clinician review",
                  caveat="Personal repeatability estimate; includes biological variability; not treatment advice")
    if any(_context(m) != _context(before) for m in [after] + repeats):
        result["warnings"].append("Supports, recording mode, engine, acquisition or metric context differ; comparison label withheld")
        result["label"] = "not comparable"
        return result
    if not stable_baseline_confirmed:
        result["label"] = "stable baseline not confirmed"
        return result
    values = np.array([m.value for m in repeats], dtype=float)
    result["excluded_sessions"] = int(np.isnan(values).sum())
    values = values[np.isfinite(values)]
    if len(values) < min_repeat_sessions:
        return result
    # Sample SD of this person's independent stable repeat-session summaries.
    mdc = float(1.96 * np.sqrt(2) * values.std(ddof=1))
    result["mdc95"] = mdc
    if not np.isfinite(before.value) or not np.isfinite(after.value):
        result["label"] = "comparison value missing"
        return result
    result["change"] = float(after.value - before.value)
    if mdc == 0:
        result["label"] = "noise estimate degenerate"
    elif source != "measured":
        result["label"] = "noise not measured"
    else:
        result["label"] = "exceeds measured noise" if abs(result["change"]) > mdc else "does not exceed measured noise"
    return result
