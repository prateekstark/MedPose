"""Metrics for aligned (frames, coordinates) angle arrays in degrees.

All functions report paired-NaN exclusions per coordinate. A missing coordinate
excludes that observation, not the other coordinates in the same frame. Metrics
use ordinary signed differences; call wrapped_difference explicitly for periodic
coordinates. No alignment, interpolation or unit conversion is performed.
"""
from typing import Mapping, NamedTuple
import warnings

import numpy as np


class MetricResult(NamedTuple):
    """values: per-coordinate metric (or framewise wrapped differences).

    excluded_frames: integer vector, one count per coordinate. All-missing
    coordinates produce NaN. An empty trial has zero exclusions and NaN metrics.
    """

    values: np.ndarray
    excluded_frames: np.ndarray


def _difference(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError("pred and true must have identical (frames, coordinates) shapes")
    if np.isinf(pred).any() or np.isinf(true).any():
        raise ValueError("Infinite angles are not supported; use NaN for missing data")
    valid = ~np.isnan(pred) & ~np.isnan(true)
    return pred - true, valid


def _mean(values, valid):
    counts = valid.sum(axis=0)
    result = np.full(values.shape[1], np.nan)
    np.divide(np.where(valid, values, 0.0).sum(axis=0), counts,
              out=result, where=counts > 0)
    return result


def _result(values, valid):
    return MetricResult(values, (~valid).sum(axis=0))


def mae(pred, true):
    """Raw per-coordinate mean absolute error and paired exclusions."""
    diff, valid = _difference(pred, true)
    return _result(_mean(np.abs(diff), valid), valid)


def median_absolute_error(pred, true):
    """Raw per-coordinate median absolute error and paired exclusions.

    PBL and B3D report medians, so compare them with this rather than with mae().
    """
    diff, valid = _difference(pred, true)
    result = np.full(diff.shape[1], np.nan)
    for coordinate in np.flatnonzero(valid.any(axis=0)):
        result[coordinate] = np.median(np.abs(diff[valid[:, coordinate], coordinate]))
    return _result(result, valid)


def rmse(pred, true):
    """Raw per-coordinate root mean squared error and paired exclusions."""
    diff, valid = _difference(pred, true)
    return _result(np.sqrt(_mean(diff ** 2, valid)), valid)


def bias(pred, true):
    """Per-coordinate mean signed error (pred minus true)."""
    diff, valid = _difference(pred, true)
    return _result(_mean(diff, valid), valid)


def bias_corrected_mae(pred, true):
    """MAE after removing each coordinate's mean error within this trial.

    This is not cohort-pooled correction and must not be labelled raw MAE.
    """
    diff, valid = _difference(pred, true)
    corrected = diff - _mean(diff, valid)
    return _result(_mean(np.abs(corrected), valid), valid)


def wrapped_difference(pred, true):
    """Framewise signed difference in [-180, 180), with missing values as NaN.

    Exactly antipodal differences use -180 by convention. Return values retain
    the input shape; excluded_frames remains a per-coordinate count vector.
    """
    diff, valid = _difference(pred, true)
    with np.errstate(invalid="ignore"):  # NaN marks missing frames and stays NaN
        return _result((diff + 180.0) % 360.0 - 180.0, valid)


def subject_bootstrap_ci(values_by_subject, n_boot=10000, ci=0.95, seed=0):
    """Equal-subject mean and percentile bootstrap confidence interval.

    Accept a mapping of subject IDs to scalar/1D observations, or a sequence
    containing one scalar/1D array per subject. Average within each subject, then
    resample subjects with replacement and average those subject means. Unequal
    trial/frame counts never give a subject extra weight. NaNs are excluded with
    a warning reporting counts by subject position. Empty/all-NaN subjects raise
    ValueError rather than being silently dropped. Returns three Python floats.
    """
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not np.isscalar(ci) or not np.isfinite(ci) or not 0 < ci < 1:
        raise ValueError("ci must be between zero and one")
    subjects = list(values_by_subject.values() if isinstance(values_by_subject, Mapping)
                    else values_by_subject)
    if not subjects:
        raise ValueError("At least one subject is required")
    means, excluded = [], []
    for subject in subjects:
        values = np.asarray(subject, dtype=float)
        if values.ndim > 1 or np.isinf(values).any():
            raise ValueError("Each subject must contain scalar or 1D finite/NaN observations")
        values = values.reshape(-1)
        valid = ~np.isnan(values)
        if not valid.any():
            raise ValueError("Every subject must have at least one non-NaN observation")
        excluded.append(int((~valid).sum()))
        means.append(float(values[valid].mean()))
    if any(excluded):
        warnings.warn("Excluded NaN observations by subject position: " + str(excluded),
                      RuntimeWarning, stacklevel=2)
    means = np.asarray(means)
    rng = np.random.default_rng(seed)
    # Loop avoids allocating n_boot * number_of_subjects indices at once.
    samples = np.empty(n_boot)
    for i in range(n_boot):
        samples[i] = means[rng.integers(0, len(means), size=len(means))].mean()
    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(means.mean()), float(low), float(high)
