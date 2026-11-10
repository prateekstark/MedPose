#!/usr/bin/env python3
"""Compare a MedPose Trial against a reference recording (gait lab or benchmark).

    python -m benchmarks.compare --prediction trial.json --reference lab.c3d \
        --lag-source "sync event" --lag 0.130 --output agreement.json
    python -m benchmarks.compare --prediction trial.json --reference ik.mot --mapping none \
        --lag-source "independent signals" --estimate-lag-on pelvis_ty --output agreement.json

The reference is resampled onto the prediction's clock at ``t + lag``. Output is
numbers only (no names, paths or video) so it can be shared when asking for help.
Results say "agreement with <reference>", because marker-based angles are a
reference, not the truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks import agreement, metrics
from benchmarks.align import LAG_SOURCES, align_reference, estimate_lag
from benchmarks.reference import c3d_channels, read_c3d, read_table, reference_events, reference_trial
from medpose.trial import Trial, load_opensim_mot


def load_reference(path, prediction, *, mapping, reference_name):
    """Return (reference Trial, mapping report, events or None)."""
    meta = dict(recording_mode=prediction.recording_mode, camera_count=prediction.camera_count,
                frame_rate=prediction.frame_rate)
    suffix = Path(path).suffix.lower()
    if suffix == ".mot":
        if mapping != "none":
            raise ValueError("A .mot reference already uses OpenSim names; pass --mapping none")
        trial = load_opensim_mot(path, engine_name=reference_name, engine_version="reference",
                                 translation_unit="m", **meta)
        return trial, dict(used=list(trial.coordinate_names)), None
    if suffix == ".c3d":
        data = read_c3d(path)
        if data["units"]["angle"] not in (None, "deg"):
            raise ValueError(f"C3D angle units are {data['units']['angle']}; expected deg")
        time, channels, events = data["time"], c3d_channels(data), reference_events(data["events"])
    elif suffix == ".csv":
        time, channels = read_table(path)
        events = None
    else:
        raise ValueError("Reference must be .c3d, .csv or .mot")
    if mapping == "none":
        names = tuple(channels)
        trial = Trial(time, names, np.column_stack([channels[n] for n in names]),
                      engine_name=reference_name, engine_version="reference", **meta)
        return trial, dict(used=list(names)), events
    trial, report = reference_trial(time, channels, mapping, reference_name=reference_name, **meta)
    return trial, report, events


def coordinate_agreement(pred, ref):
    """Per-coordinate MAE, RMSE, bias and offset-removed RMSE on aligned Trials."""
    rows = {}
    mae, rmse, bias = metrics.mae(pred.values, ref.values), metrics.rmse(pred.values, ref.values), metrics.bias(pred.values, ref.values)
    for j, name in enumerate(pred.coordinate_names):
        diff = pred.values[:, j] - ref.values[:, j]
        valid = np.isfinite(diff)
        removed = float(np.sqrt(np.mean((diff[valid] - diff[valid].mean()) ** 2))) if valid.any() else None
        rows[name] = dict(unit=pred.units[j], frames=int(len(diff)), compared_frames=int(valid.sum()),
                          coverage=float(valid.mean()) if len(diff) else None,
                          mae=None if not valid.any() else float(mae.values[j]),
                          rmse=None if not valid.any() else float(rmse.values[j]),
                          bias=None if not valid.any() else float(bias.values[j]),
                          rmse_offset_removed=removed)
    return rows


def parameter_agreement(pred, ref, *, task):
    """Discrete parameters on the prediction's gait cycles, computed in both systems.

    Using the same cycles isolates angle agreement from event-detection error.
    Cycles within one trial are not independent subjects: pool several people with
    agreement.bland_altman_repeated for publishable limits.
    """
    from medpose.analyse import analyse_walk
    from medpose.clinical.parameters import gait_parameters
    from medpose.engine.opencap_trial import planes_for

    gait, pred_params = analyse_walk(pred, task=task)
    # ``ref`` is already on the prediction's clock, so the prediction's cycle frames index it directly.
    ref_params = gait_parameters(ref, gait, planes=planes_for(ref.coordinate_names), task=task)
    out = {}
    for name, entry in pred_params["parameters"].items():
        out[name] = {}
        for side in ("left", "right"):
            p = entry["sides"][side]["per_cycle"]
            r = ref_params["parameters"][name]["sides"][side]["per_cycle"]
            if not p or not r:
                continue
            p = np.array([np.nan if v is None else v for v in p])
            r = np.array([np.nan if v is None else v for v in r])
            result = agreement.bland_altman(p, r)
            result["note"] = "cycles from one trial, not independent subjects"
            out[name][side] = result
    return gait, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prediction", required=True, type=Path, help="MedPose Trial JSON")
    parser.add_argument("--reference", required=True, type=Path, help=".c3d, .csv (with time column) or .mot")
    parser.add_argument("--reference-name", default="marker-based reference",
                        help="What the reference is, e.g. 'Vicon Plug-in Gait (hospital lab)'")
    parser.add_argument("--mapping", default="plugin_gait",
                        help="Convention mapping name/path in benchmarks/conventions, or 'none' if names match")
    parser.add_argument("--lag-source", required=True, choices=sorted(LAG_SOURCES),
                        help="How the time offset was obtained; reported with the results")
    parser.add_argument("--lag", type=float, help="Known offset in seconds (reference time = prediction time + lag)")
    parser.add_argument("--estimate-lag-on", nargs="+", metavar="COORDINATE",
                        help="Estimate the offset by correlation on these coordinates")
    parser.add_argument("--max-lag", type=float, default=1.0)
    parser.add_argument("--lag-step", type=float, default=0.002)
    parser.add_argument("--max-gap", type=float, default=0.05, help="Longest reference gap to interpolate (s)")
    parser.add_argument("--event-tolerance", type=float, default=0.15)
    parser.add_argument("--task", default="walking")
    parser.add_argument("--no-parameters", action="store_true", help="Skip gait-cycle parameter agreement")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if (args.lag is None) == (args.estimate_lag_on is None):
            raise ValueError("Give exactly one of --lag or --estimate-lag-on")
        if args.lag is not None and args.lag_source not in {"sync event", "other trials", "none"}:
            raise ValueError("A fixed --lag must come from a sync event, other trials, or be 'none' (0)")
        if args.estimate_lag_on is not None and args.lag_source not in {"independent signals", "scored signals"}:
            raise ValueError("An estimated lag's source is 'independent signals' or 'scored signals'")
        if args.output and args.output.exists():
            raise ValueError(f"Refusing to overwrite {args.output}")
        prediction = Trial.load_json(args.prediction)
        reference, mapping_report, ref_events = load_reference(args.reference, prediction, mapping=args.mapping,
                                                               reference_name=args.reference_name)
        lag_details = None
        if args.estimate_lag_on:
            names = args.estimate_lag_on
            for n in names:
                if n not in prediction.coordinate_names or n not in reference.coordinate_names:
                    raise ValueError(f"Lag coordinate {n} missing from prediction or reference")
            p = prediction.values[:, [prediction.coordinate_names.index(n) for n in names]]
            r = reference.values[:, [reference.coordinate_names.index(n) for n in names]]
            lag_details = estimate_lag(prediction.time, p, reference.time, r, max_lag_s=args.max_lag,
                                       step_s=args.lag_step, max_gap_s=args.max_gap)
            lag = lag_details["lag_s"]
            scored = set(names) & set(mapping_report.get("used", []) + mapping_report.get("caution", []))
            if scored and args.lag_source == "independent signals":
                raise ValueError("Lag coordinates are also scored; use --lag-source 'scored signals' to say so")
        else:
            lag = args.lag
        pred, ref = align_reference(prediction, reference, lag_s=lag, max_gap_s=args.max_gap)
        result = dict(reference=args.reference_name, comparison="agreement with reference, not accuracy against truth",
                      lag_s=lag, lag_source=args.lag_source, lag_search=lag_details, mapping=mapping_report,
                      coordinates=coordinate_agreement(pred, ref))
        if args.lag_source == "scored signals":
            result["warning"] = "Offset was tuned on scored angles; errors are optimistic"
        if not args.no_parameters:
            try:
                gait, result["parameters"] = parameter_agreement(prediction, ref, task=args.task)
            except ValueError as exc:
                result["parameters"] = dict(status=f"not computed: {exc}")
                gait = None
            if ref_events is not None and gait is not None:
                result["events"] = {}
                for side in ("left", "right"):
                    ev = gait["sides"][side]["events"]
                    for key, frames in (("foot_strike", ev["initial_contacts"]), ("foot_off", ev["departures"])):
                        predicted = [float(prediction.time[i]) + lag for i in frames]
                        if ref_events[side][key]:
                            result["events"][f"{side}_{key}"] = agreement.event_agreement(
                                predicted, ref_events[side][key], tolerance_s=args.event_tolerance)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    text = json.dumps(result, indent=1, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    angles = {k: v for k, v in result["coordinates"].items() if v["unit"] == "deg" and v["rmse"] is not None}
    print(f"Lag {lag:+.3f} s ({args.lag_source}). Agreement with {args.reference_name}:")
    for name, row in angles.items():
        print(f"  {name:22s} RMSE {row['rmse']:5.1f}  offset-removed {row['rmse_offset_removed']:5.1f}  "
              f"bias {row['bias']:+5.1f} deg  coverage {row['coverage']:.0%}")
    if not args.output:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
