"""Offline benchmark aggregation with explicit statistic/reference/correction context."""
import json
from pathlib import Path

import numpy as np

from benchmarks import metrics
from medpose.errors import ErrorTable, SOURCES
from medpose.trial import Trial

STATISTICS = {"mae": metrics.mae, "rmse": metrics.rmse,
              "median_absolute_error": metrics.median_absolute_error}
AGGREGATION = "equal-subject mean of within-subject mean trial statistics"


def benchmark_results(trials_by_subject, *, method, planes, task, reference_system,
                      source, origin, statistic="mae", wrapped_coordinates=(), n_boot=10000, seed=0):
    """Mapping of anonymous subject keys to (prediction Trial, reference Trial) pairs.

    Keys are only grouping tokens and are not exported. Predictions and reference
    must already be exactly time-aligned. No resampling or cohort offset is fitted.
    Both raw and within-trial-mean-offset-corrected results are retained distinctly.
    """
    for text in (method, task, reference_system, origin):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Method, task, reference and origin are required")
    if source not in SOURCES or statistic not in STATISTICS:
        raise ValueError("Unknown source or statistic")
    if not isinstance(trials_by_subject, dict):
        raise ValueError("Provide a subject-grouped mapping")
    if not trials_by_subject:
        return []
    first, signature = None, None
    values, counts = {}, {}
    for subject_index, pairs in enumerate(trials_by_subject.values()):
        for pred, true in pairs:
            if not isinstance(pred, Trial) or not isinstance(true, Trial):
                raise ValueError("Benchmark inputs must be Trials")
            if pred.coordinate_names != true.coordinate_names or not np.array_equal(pred.time, true.time):
                raise ValueError("Predictions/reference must have matching coordinates and aligned timestamps")
            current = (pred.recording_mode, pred.coordinate_names, pred.units)
            if signature is not None and signature != current:
                raise ValueError("One benchmark batch must have the same mode, coordinates and units")
            first, signature = pred, current
            if any(n not in planes for n in pred.coordinate_names):
                raise ValueError("Every coordinate needs an explicit plane")
            if any(n not in pred.coordinate_names for n in wrapped_coordinates):
                raise ValueError("Unknown wrapped coordinate")
            difference = pred.values - true.values
            for j, name in enumerate(pred.coordinate_names):
                if name in wrapped_coordinates:
                    if pred.units[j] != "deg":
                        raise ValueError("Cannot wrap translations")
                    difference[:, j] = metrics.wrapped_difference(pred.values[:, j:j+1], true.values[:, j:j+1]).values[:, 0]
            zeros = np.zeros_like(difference)
            raw = STATISTICS[statistic](difference, zeros)
            offsets = metrics.bias(difference, zeros).values
            corrected = (metrics.bias_corrected_mae(difference, zeros) if statistic == "mae"
                         else STATISTICS[statistic](difference - offsets, zeros))
            for j, name in enumerate(pred.coordinate_names):
                counts.setdefault(name, [0, 0])
                counts[name][0] += len(pred.time)
                counts[name][1] += int(raw.excluded_frames[j])
                for correction, value in (("none", raw.values[j]), ("within-trial mean offset", corrected.values[j])):
                    key = (name, correction)
                    values.setdefault(key, {})
                    if np.isfinite(value):
                        values[key].setdefault(subject_index, []).append(float(value))
    if first is None:
        return []
    result = []
    for (name, correction), subjects in values.items():
        point, ci = None, None
        if subjects:
            point, low, high = metrics.subject_bootstrap_ci(subjects, n_boot=n_boot, seed=seed)
            ci = [low, high] if len(subjects) >= 2 else None
        total, excluded = counts[name]
        result.append(dict(method=method, mode=first.recording_mode, coordinate=name, plane=planes[name],
            task=task, unit=first.units[first.coordinate_names.index(name)], statistic=statistic,
            aggregation=AGGREGATION, reference_system=reference_system, correction=correction,
            wrapped=name in wrapped_coordinates, value=point, ci95=ci,
            subjects_used=len(subjects), subjects_excluded=len(trials_by_subject)-len(subjects),
            total_frames=total, excluded_frames=excluded, coverage=(total-excluded)/total if total else None,
            source=source, origin=origin, run_status="reproduced", n_boot=n_boot, seed=seed))
    return result


def published_result(*, method, mode, coordinate, plane, task, unit, statistic,
                     aggregation, reference_system, correction, value, source, origin, wrapped=False):
    """Record a cited result without pretending a local run exists."""
    row = dict(method=method, mode=mode, coordinate=coordinate, plane=plane, task=task, unit=unit,
               statistic=statistic, aggregation=aggregation, reference_system=reference_system,
               correction=correction, value=value, source=source, origin=origin, wrapped=wrapped,
               run_status="published", ci95=None, coverage=None, subjects_used=None,
               subjects_excluded=None, total_frames=None, excluded_frames=None)
    _validate(row)
    return row


def _validate(row):
    required = ("method", "mode", "coordinate", "plane", "task", "unit", "statistic", "aggregation",
                "reference_system", "correction", "source", "origin", "run_status")
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
        raise ValueError("Benchmark result lacks comparison context")
    if row["source"] not in SOURCES or row["run_status"] not in {"published", "reproduced"}:
        raise ValueError("Invalid benchmark provenance")
    value = row.get("value")
    if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or value < 0):
        raise ValueError("Error value must be finite nonnegative or missing")


def compare_results(rows, *, mismatch_label=None):
    """Refuse incompatible comparisons unless prominently and explicitly labelled."""
    rows = list(rows)
    for row in rows:
        _validate(row)
    keys = ("statistic", "aggregation", "reference_system", "correction", "wrapped", "coordinate", "plane", "task", "unit")
    differences = [key for key in keys if len({row.get(key) for row in rows}) > 1]
    if differences and (not isinstance(mismatch_label, str) or not mismatch_label.strip()):
        raise ValueError("Incompatible benchmark comparison: " + ", ".join(differences))
    return dict(rows=rows, mismatch_fields=differences, comparison_label=mismatch_label if differences else "matched context",
                warning="Not a like-for-like comparison" if differences else None)


def export_error_table(rows, path):
    """Export only raw run estimates; publication summaries need explicit manual review.

    A2 has no method/reference dimension: refuse mixing them or duplicate keys.
    CI, aggregation and coverage are retained in origin; a scalar is not a
    per-person uncertainty interval. Corrected errors are never silently exported.
    """
    rows = list(rows)
    for row in rows:
        _validate(row)
        if row["correction"] != "none" or row["run_status"] != "reproduced":
            raise ValueError("Export requires raw reproduced benchmark rows")
    for key in ("method", "statistic", "aggregation", "reference_system", "source"):
        if len({row[key] for row in rows}) > 1:
            raise ValueError("Error-table export cannot mix " + key)
    entries = []
    for row in rows:
        if row["value"] is None:
            continue
        context = {key: row.get(key) for key in ("method", "statistic", "aggregation", "reference_system", "coverage", "ci95", "origin", "wrapped")}
        entries.append({key: row[key] for key in ("mode", "coordinate", "plane", "task", "value", "source", "unit")})
        entries[-1]["origin"] = json.dumps(context, allow_nan=False, sort_keys=True)
    ErrorTable(entries)  # Check units, provenance and collisions before writing.
    Path(path).write_text(json.dumps(dict(schema_version=1, entries=entries), indent=2, allow_nan=False) + "\n", encoding="utf-8")
