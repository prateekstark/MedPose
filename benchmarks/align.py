"""Put a prediction and a reference recording on one clock before comparing them.

Video runs at 30-60 fps and marker systems at 100-200 Hz, with an unknown start
offset between the two clocks. Nothing here hides that: resampling never bridges
gaps, and the time offset ("lag") is always reported next to the result.

Lag convention: the reference value matching prediction time ``t`` is the
reference sample at ``t + lag_s``. A positive lag means the reference clock reads
later than the prediction clock for the same instant.

Choosing the lag that best matches the very angles you then score makes the
error look smaller than it is. Prefer, in order: (1) a shared sync event (LED
flash, clap, force-plate strike) measured independently; (2) a lag estimated on
signals you do not score, such as pelvis translation; (3) a lag estimated on
other trials of the same session and frozen. If you must estimate it from the
scored angles, say so wherever the numbers go (``lag_source`` below).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from medpose.trial import Trial

LAG_SOURCES = {"sync event", "independent signals", "other trials", "scored signals", "none"}


def resample(time, values, target_time, *, max_gap_s):
    """Linearly interpolate (frames, channels) onto ``target_time``.

    A target sample is NaN when it lies outside the source time range, when the two
    bracketing source samples are more than ``max_gap_s`` apart, or when either of
    them is NaN. Exact source timestamps return the source value unchanged.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target_time, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if time.ndim != 1 or len(time) != len(values) or len(time) < 2:
        raise ValueError("Need at least two source samples with one time per row")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Source time must be finite and strictly increasing")
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError("Target time must be a finite 1D array")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    out = np.full((len(target), values.shape[1]), np.nan)
    inside = (target >= time[0]) & (target <= time[-1])
    index = np.searchsorted(time, target, side="left")
    hit = np.minimum(index, len(time) - 1)
    exact = inside & (time[hit] == target)
    # Exact hits use the sample itself, so a NaN neighbour cannot erase a real value.
    out[exact] = values[hit[exact]]
    right = np.clip(index, 1, len(time) - 1)
    left = right - 1
    between = inside & ~exact
    l, r = left[between], right[between]
    gap_ok = (time[r] - time[l]) <= max_gap_s
    weight = ((target[between] - time[l]) / (time[r] - time[l]))[:, None]
    interpolated = values[l] * (1 - weight) + values[r] * weight
    interpolated[~gap_ok] = np.nan
    out[between] = interpolated  # NaN propagates from either neighbour.
    return out


def estimate_lag(pred_time, pred, ref_time, ref, *, max_lag_s, step_s, max_gap_s,
                 min_overlap_fraction=0.5):
    """Search lags in [-max_lag_s, max_lag_s] for the best normalised correlation.

    ``pred`` and ``ref`` are (frames, channels) with matching channel order; each
    channel is z-scored over the overlapping samples so that no channel dominates
    by its units. Returns the lag, its correlation, the full search curve and
    warnings when the best lag sits on the search edge or the peak is flat.
    """
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    if ref.ndim == 1:
        ref = ref[:, None]
    if pred.shape[1] != ref.shape[1]:
        raise ValueError("Prediction and reference need the same channels for lag search")
    if not (np.isfinite(max_lag_s) and max_lag_s > 0 and np.isfinite(step_s) and 0 < step_s <= max_lag_s):
        raise ValueError("Require 0 < step_s <= max_lag_s")
    if not 0 < min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must lie in (0, 1]")
    pred_time = np.asarray(pred_time, dtype=float)
    lags = np.arange(-max_lag_s, max_lag_s + step_s / 2, step_s)
    scores = np.full(len(lags), np.nan)
    pred_valid = np.isfinite(pred).all(axis=1)
    if not pred_valid.any():
        raise ValueError("Prediction has no complete samples for lag search")
    for i, lag in enumerate(lags):
        shifted = resample(ref_time, ref, pred_time + lag, max_gap_s=max_gap_s)
        valid = pred_valid & np.isfinite(shifted).all(axis=1)
        if valid.sum() < max(3, min_overlap_fraction * pred_valid.sum()):
            continue
        a, b = pred[valid], shifted[valid]
        sa, sb = a.std(axis=0), b.std(axis=0)
        if np.any(sa == 0) or np.any(sb == 0):
            continue
        scores[i] = float(np.mean(((a - a.mean(0)) / sa) * ((b - b.mean(0)) / sb)))
    if not np.isfinite(scores).any():
        raise ValueError("No lag had enough overlapping, varying samples; widen the search or check the data")
    best = int(np.nanargmax(scores))
    warnings = []
    if best in (0, len(lags) - 1):
        warnings.append("Best lag is on the search edge; the true offset may lie outside the window")
    # Neighbouring lags always score almost as well; compare against other local peaks.
    filled = np.where(np.isfinite(scores), scores, -np.inf)
    peaks, _ = find_peaks(np.r_[-np.inf, filled, -np.inf])
    others = [filled[p - 1] for p in peaks if p - 1 != best]
    if others and max(others) > scores[best] - 0.05:
        warnings.append("Another lag correlates almost as well; the offset is ambiguous, prefer a sync event")
    if scores[best] < 0.8:
        warnings.append("Best correlation is below 0.8; signals may not correspond")
    return dict(lag_s=float(lags[best]), correlation=float(scores[best]), step_s=float(step_s),
                max_lag_s=float(max_lag_s), lags_s=lags.tolist(),
                correlations=[None if not np.isfinite(s) else float(s) for s in scores],
                warnings=warnings)


def align_reference(prediction, reference, *, lag_s, max_gap_s, coordinates=None):
    """Return (prediction, reference) Trials on the prediction's timestamps.

    Only coordinates present in both are kept (or ``coordinates`` if given, which
    must all be present). Reference values are resampled at ``time + lag_s``;
    reference confidence is not carried over. The returned reference Trial copies
    the prediction's acquisition metadata so benchmark code can pair them, with the
    reference's engine name and version kept for provenance.
    """
    if not isinstance(prediction, Trial) or not isinstance(reference, Trial):
        raise ValueError("Provide prediction and reference Trials")
    if not np.isfinite(lag_s):
        raise ValueError("lag_s must be finite")
    shared = [n for n in prediction.coordinate_names if n in reference.coordinate_names]
    if coordinates is not None:
        missing = [n for n in coordinates if n not in shared]
        if missing:
            raise ValueError("Coordinates missing from prediction or reference: " + ", ".join(missing))
        shared = list(coordinates)
    if not shared:
        raise ValueError("Prediction and reference share no coordinates; map conventions first")
    for name in shared:
        if prediction.units[prediction.coordinate_names.index(name)] != reference.units[reference.coordinate_names.index(name)]:
            raise ValueError(f"Unit mismatch for {name}")
    p_idx = [prediction.coordinate_names.index(n) for n in shared]
    r_idx = [reference.coordinate_names.index(n) for n in shared]
    ref_values = resample(reference.time, reference.values[:, r_idx], prediction.time + lag_s, max_gap_s=max_gap_s)
    common = dict(recording_mode=prediction.recording_mode, camera_count=prediction.camera_count,
                  frame_rate=prediction.frame_rate, supports=prediction.supports)
    pred_conf = None if prediction.confidence is None else prediction.confidence[:, p_idx]
    pred_out = Trial(prediction.time, tuple(shared), prediction.values[:, p_idx],
                     engine_name=prediction.engine_name, engine_version=prediction.engine_version,
                     confidence=pred_conf, **common)
    ref_out = Trial(prediction.time, tuple(shared), ref_values, engine_name=reference.engine_name,
                    engine_version=reference.engine_version, **common)
    return pred_out, ref_out
