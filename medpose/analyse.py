#!/usr/bin/env python3
"""Analyse a walking Trial: gait events, cycle curves, parameters and a report.

    python -m medpose.analyse run trial.json --output-dir out/
    python -m medpose.analyse run trial.json --previous older.json --normative ref.json --output-dir out/
    python -m medpose.analyse normative td1.json td2.json ... --source "..." --output ref.json

Writes ``gait.json`` (numbers only) and ``report.html``. Run on patient data from
your own terminal; the outputs contain no names but still describe one person's
movement, so keep them under ``patient_data/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medpose.clinical.events import progression_direction, zeni_events, ZENI_ORIGIN, VALIDATION_NOTE
from medpose.clinical.gait import analyse_gait
from medpose.clinical.parameters import (build_normative, gait_parameters, gait_profile_score,
                                         load_normative, save_normative)
from medpose.engine.opencap_trial import planes_for
from medpose.report.gait_report import render_gait_report
from medpose.trial import Trial, _unique_keys

CONFIG = Path(__file__).with_name("config") / "gait_events.json"
DEFAULT_CONVENTION = "OpenCap Core LaiUhlrich2022"
GPS_VARIABLES = ("pelvis_tilt", "pelvis_list", "pelvis_rotation", "hip_flexion", "hip_adduction",
                 "hip_rotation", "knee_angle", "ankle_angle")


def load_event_settings(path=CONFIG):
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    expected = {"schema_version", "min_step_interval_s", "max_gap_s", "min_distance_m", "min_straightness",
                "min_confidence", "status", "origin"}
    if set(data) != expected or data["schema_version"] != 1:
        raise ValueError("Malformed gait event settings")
    if data["status"] != "provisional: needs clinician review":
        raise ValueError("Gait event settings must stay labelled provisional until validated")
    return data


def analyse_walk(trial, *, task="walking", settings=None):
    """Events (Zeni), gait cycles and parameters for one straight walking pass."""
    settings = load_event_settings() if settings is None else settings
    direction, path = progression_direction(trial, min_distance_m=settings["min_distance_m"],
                                            min_straightness=settings["min_straightness"])
    events = {side: zeni_events(trial, side, direction=direction,
                                min_step_interval_s=settings["min_step_interval_s"],
                                max_gap_s=settings["max_gap_s"]) for side in ("left", "right")}
    planes = planes_for(trial.coordinate_names)
    gait = analyse_gait(trial, None, planes=planes, task=task, events=events,
                        progression_direction=direction, min_confidence=settings["min_confidence"])
    gait["path"] = path
    gait["event_settings"] = settings
    parameters = gait_parameters(trial, gait, planes=planes, task=task, min_confidence=settings["min_confidence"])
    return gait, parameters


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def run(args):
    out = args.output_dir
    targets = [out / "gait.json", out / "report.html"]
    if any(t.exists() for t in targets):
        raise ValueError(f"Refusing to overwrite results in {out}")
    trial = Trial.load_json(args.trial)
    gait, parameters = analyse_walk(trial, task=args.task)
    previous_gait = previous_parameters = None
    if args.previous:
        previous = Trial.load_json(args.previous)
        if (previous.recording_mode, sorted(previous.supports), previous.engine_name) != \
                (trial.recording_mode, sorted(trial.supports), trial.engine_name):
            print("Warning: previous session used a different mode, supports or engine; compare with care")
        previous_gait, previous_parameters = analyse_walk(previous, task=args.task)
    normative = gps = None
    if args.normative:
        normative = load_normative(args.normative)
        gps = gait_profile_score(gait, normative, convention=args.convention)
    html = render_gait_report(trial, gait, parameters, task=args.task,
                              events_method=f"{ZENI_ORIGIN}. {VALIDATION_NOTE}", gps=gps, normative=normative,
                              previous_gait=previous_gait, previous_parameters=previous_parameters)
    out.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(dict(gait=gait, parameters=parameters, gait_profile=gps),
                                     indent=1, default=_json, allow_nan=False) + "\n", encoding="utf-8")
    targets[1].write_text(html, encoding="utf-8")
    cycles = {s: len(gait["sides"][s]["events"]["cycles"]) for s in ("left", "right")}
    print(f"Cycles: left {cycles['left']}, right {cycles['right']}. Wrote {targets[0]} and {targets[1]}")
    if not any(cycles.values()):
        print("No complete cycles: check the pass is straight, long enough, and the feet were tracked")
    return 0


def normative(args):
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    gaits = [analyse_walk(Trial.load_json(p), task=args.task)[0] for p in args.trials]
    data = build_normative(gaits, variables=args.variables, convention=args.convention, source=args.source)
    save_normative(data, args.output)
    print(f"Wrote {args.output} from {len(gaits)} trials ({data['samples']} samples per curve)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Analyse one walking trial and write gait.json and report.html")
    r.add_argument("trial", type=Path)
    r.add_argument("--output-dir", required=True, type=Path)
    r.add_argument("--task", default="walking")
    r.add_argument("--previous", type=Path, help="An earlier Trial to overlay (same setup)")
    r.add_argument("--normative", type=Path, help="Reference curves made with 'normative' from the same pipeline")
    r.add_argument("--convention", default=DEFAULT_CONVENTION)
    n = sub.add_parser("normative", help="Build reference curves from typically developing Trials")
    n.add_argument("trials", nargs="+", type=Path)
    n.add_argument("--source", required=True, help="Where the recordings came from (no names)")
    n.add_argument("--output", required=True, type=Path)
    n.add_argument("--task", default="walking")
    n.add_argument("--convention", default=DEFAULT_CONVENTION)
    n.add_argument("--variables", nargs="+", default=list(GPS_VARIABLES))
    args = parser.parse_args()
    try:
        return run(args) if args.command == "run" else normative(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
