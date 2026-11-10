"""Gait events from foot and pelvis positions (Zeni et al. 2008 coordinate method).

Zeni, Richards & Higginson (2008), Gait & Posture 27:710-714, PMC2384115:
heel strike is a local maximum of (heel - sacrum) along the walking direction and
toe-off a local minimum of (toe - sacrum); "the mean value of this curve is used
to set a threshold" for peaks and valleys. Overground walking is handled by
subtracting the sacrum, so the person "appears to be walking in place".

Validated (against force plates) in healthy adults, adults with MS and adults after
stroke, at 60 Hz. **Not validated in children with CP, in toe-walking, or in
walker-assisted gait.** In equinus the first contact may be the forefoot, so the
"heel strike" here means the foot's most-forward moment, which can differ from
true initial contact. Check events against video before trusting cycle timing.

This module uses the pelvis origin (``pelvis_tx/ty/tz``) in place of the sacrum
marker; a constant offset between the two does not move the peaks.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

ZENI_ORIGIN = "Zeni, Richards & Higginson 2008, Gait Posture 27:710-714 (coordinate method)"
VALIDATION_NOTE = ("Validated against force plates in healthy, MS and stroke adults; not validated in "
                   "children with CP, toe walking or walker-assisted gait")
PELVIS = ("pelvis_tx", "pelvis_ty", "pelvis_tz")


def progression_direction(trial, *, min_distance_m, min_straightness):
    """Unit walking direction in the ground plane from net pelvis displacement.

    Refuses trials that do not travel far enough or that turn: straightness is net
    displacement / path length in the ground plane. Returns (direction, details).
    """
    feet = trial.foot_positions
    if feet is None:
        raise ValueError("Trial has no foot positions (and so no ground plane)")
    if any(n not in trial.coordinate_names for n in PELVIS):
        raise ValueError("Trial lacks pelvis translations")
    pelvis = trial.values[:, [trial.coordinate_names.index(n) for n in PELVIS]]
    valid = np.isfinite(pelvis).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("Not enough pelvis samples to find the walking direction")
    p = pelvis[valid]
    flat = p - np.outer(p @ feet.normal, feet.normal)
    net = flat[-1] - flat[0]
    distance = float(np.linalg.norm(net))
    path = float(np.linalg.norm(np.diff(flat, axis=0), axis=1).sum())
    straightness = distance / path if path > 0 else 0.0
    details = dict(net_distance_m=distance, path_length_m=path, straightness=straightness)
    if distance < min_distance_m:
        raise ValueError(f"Pelvis travelled {distance:.2f} m, less than {min_distance_m} m; not a walking pass")
    if straightness < min_straightness:
        raise ValueError(f"Path straightness {straightness:.2f} < {min_straightness}; trim turns before analysis")
    return net / distance, details


def _segments(valid):
    """Half-open [start, stop) runs of True."""
    edges = np.diff(np.r_[0, valid.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def zeni_events(trial, side, *, direction, min_step_interval_s, max_gap_s):
    """Detect heel strikes and toe-offs for one side and pair them into cycles.

    ``min_step_interval_s`` is the shortest allowed time between two events of the
    same type on the same foot (a cycle can't be shorter). It is a segmentation
    setting the caller must choose, not a clinical threshold. Events are searched
    only within runs of complete data no longer interrupted than ``max_gap_s``.
    Returns the same structure as ``gait.detect_contacts``.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if not (np.isfinite(min_step_interval_s) and min_step_interval_s > 0 and np.isfinite(max_gap_s) and max_gap_s > 0):
        raise ValueError("min_step_interval_s and max_gap_s must be positive")
    direction = np.asarray(direction, dtype=float)
    result = dict(status="detection failed", reason="foot positions unavailable", method=ZENI_ORIGIN,
                  validation=VALIDATION_NOTE, initial_contacts=[], departures=[], cycles=[],
                  low_confidence_frames=[], unknown_confidence_frames=[], invalid_frames=[],
                  config=dict(min_step_interval_s=min_step_interval_s, max_gap_s=max_gap_s,
                              direction=direction.tolist()),
                  source="provisional", error_label="error not yet measured")
    feet = trial.foot_positions
    if feet is None or getattr(feet, side) is None:
        return result
    if any(n not in trial.coordinate_names for n in PELVIS):
        result["reason"] = "pelvis translations unavailable"
        return result
    pelvis = trial.values[:, [trial.coordinate_names.index(n) for n in PELVIS]]
    points = getattr(feet, side)
    heel = (points[:, 0] - pelvis) @ direction
    toe = (points[:, 1] - pelvis) @ direction
    valid = np.isfinite(heel) & np.isfinite(toe)
    confidence = getattr(feet, side + "_confidence")
    if confidence is None:
        result["unknown_confidence_frames"] = list(range(len(points)))
    result["invalid_frames"] = np.flatnonzero(~valid).tolist()
    dt = np.diff(trial.time)
    starts, ends = [], []
    for a, b in _segments(valid):
        # Split further wherever the clock jumps by more than max_gap_s.
        cuts = [a] + [i + 1 for i in range(a, b - 1) if dt[i] > max_gap_s] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            if e - s < 3:
                continue
            spacing = float(np.median(dt[s:e - 1]))
            distance = max(1, int(round(min_step_interval_s / spacing)))
            # Zeni: the mean of the curve is the threshold for peaks (HS) and valleys (TO).
            hs, _ = find_peaks(heel[s:e], height=heel[s:e].mean(), distance=distance)
            to, _ = find_peaks(-toe[s:e], height=-toe[s:e].mean(), distance=distance)
            starts.extend((hs + s).tolist())
            ends.extend((to + s).tolist())
    starts, ends = np.array(sorted(starts), dtype=int), np.array(sorted(ends), dtype=int)
    result["initial_contacts"], result["departures"] = starts.tolist(), ends.tolist()
    for start, stop in zip(starts[:-1], starts[1:]):
        offs = ends[(ends > start) & (ends < stop)]
        if len(offs) != 1 or not valid[start:stop + 1].all() or np.any(dt[start:stop] > max_gap_s):
            continue
        result["cycles"].append(dict(start=int(start), departure=int(offs[0]), stop=int(stop)))
    if result["cycles"]:
        result["status"] = "candidate cycles"
        result["reason"] = "Zeni coordinate method; " + VALIDATION_NOTE
    else:
        result["reason"] = "no complete cycle with exactly one toe-off between heel strikes"
    return result


def event_times(trial, events):
    """Seconds of each detected event, for comparison with reference events."""
    return dict(foot_strike=[float(trial.time[i]) for i in events["initial_contacts"]],
                foot_off=[float(trial.time[i]) for i in events["departures"]])
