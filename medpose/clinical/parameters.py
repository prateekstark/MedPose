"""Discrete gait parameters per cycle, and a GPS-style deviation from normative curves.

Discrete parameters are the numbers gait clinicians usually read off kinematic
graphs (knee flexion at initial contact, minimum knee flexion in stance, peak knee
flexion in swing, peak ankle dorsiflexion in stance...). Here they are descriptive
measurements with an error label, not targets or diagnoses. Angles keep the engine's
model conventions: OpenSim's ankle 0 deg is the model's neutral, not the clinical
foot-to-shank zero, so absolute ankle values are not interchangeable with a lab's.

The Gait Profile Score (Baker et al. 2009, Gait Posture 30:265-269) is the RMS
difference between a person's mean cycle curves and a typically developing
reference, over nine specific kinematic variables including foot progression,
which OpenSim coordinates do not provide. So this module computes a *GPS-style*
score over whatever variables you choose. It is not comparable to published GPS
values, and the GPS minimal clinically important difference of 1.6 deg (Baker et al.
2012, Gait Posture 35:612-615) does not transfer. Its use is tracking one person
over time against a fixed reference made with the same pipeline and conventions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medpose.errors import ErrorTable
from medpose.trial import _unique_keys

# name, coordinate base, phase, statistic. Phases: initial contact (first frame of
# the cycle), stance (initial contact..toe-off), swing (toe-off..next contact), cycle.
PARAMETERS = (
    ("knee_flexion_at_initial_contact", "knee_angle", "initial contact", "value"),
    ("knee_flexion_min_stance", "knee_angle", "stance", "minimum"),
    ("knee_flexion_peak_swing", "knee_angle", "swing", "maximum"),
    ("knee_flexion_range", "knee_angle", "cycle", "range"),
    ("hip_flexion_min_stance", "hip_flexion", "stance", "minimum"),
    ("hip_flexion_peak", "hip_flexion", "cycle", "maximum"),
    ("ankle_angle_at_initial_contact", "ankle_angle", "initial contact", "value"),
    ("ankle_dorsiflexion_peak_stance", "ankle_angle", "stance", "maximum"),
    ("ankle_angle_min_swing", "ankle_angle", "swing", "minimum"),
)
DESCRIPTIONS = {
    "knee_flexion_at_initial_contact": "Knee flexion when the foot strikes",
    "knee_flexion_min_stance": "Straightest knee during stance (high values: knee stays bent)",
    "knee_flexion_peak_swing": "Most knee bend during swing (low values: stiff knee)",
    "knee_flexion_range": "Knee range of motion over the cycle",
    "hip_flexion_min_stance": "Most hip extension during stance",
    "hip_flexion_peak": "Most hip flexion over the cycle",
    "ankle_angle_at_initial_contact": "Ankle angle (dorsiflexion positive) at foot strike; foot and brace together",
    "ankle_dorsiflexion_peak_stance": "Most ankle dorsiflexion during stance; foot and brace together",
    "ankle_angle_min_swing": "Lowest ankle angle (most plantarflexion) in swing; foot and brace together",
}
REDUCERS = {"minimum": np.min, "maximum": np.max, "range": np.ptp}
GPS_CAVEAT = ("GPS-style score over the listed variables only; not the published Gait Profile Score, "
              "not comparable with published values, and the 1.6 deg GPS MCID does not apply")


def _summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(per_cycle=[float(v) if np.isfinite(v) else None for v in values],
                mean=float(finite.mean()) if len(finite) else None,
                sd=float(finite.std(ddof=1)) if len(finite) > 1 else None, count=int(len(finite)))


def _window(cycle, phase):
    start, off, stop = cycle["start"], cycle["departure"], cycle["stop"]
    return {"initial contact": (start, start), "stance": (start, off), "swing": (off, stop),
            "cycle": (start, stop)}[phase]


def gait_parameters(trial, gait, *, planes, task="walking", errors=None, min_confidence=None):
    """Per-cycle discrete parameters for each side, from ``analyse_gait`` output.

    A cycle contributes nothing for a parameter if any value in its window is
    missing, or below ``min_confidence`` when confidence exists and a cutoff is given.
    """
    errors = ErrorTable.load() if errors is None else errors
    output = dict(source="provisional", conventions=f"{trial.engine_name} model coordinates",
                  parameters={})
    for name, base, phase, statistic in PARAMETERS:
        entry = dict(coordinate=base, phase=phase, statistic=statistic, unit="deg",
                     description=DESCRIPTIONS[name], sides={}, left_minus_right=None)
        for side in ("left", "right"):
            coordinate = f"{base}_{side[0]}"
            if coordinate not in trial.coordinate_names:
                entry["sides"][side] = dict(status="coordinate unavailable", **_summary([]))
                continue
            j = trial.coordinate_names.index(coordinate)
            values = []
            for cycle in gait["sides"][side]["events"]["cycles"]:
                a, b = _window(cycle, phase)
                window = trial.values[a:b + 1, j]
                ok = np.isfinite(window).all()
                if ok and trial.confidence is not None and min_confidence is not None:
                    conf = trial.confidence[a:b + 1, j]
                    ok = bool(np.isfinite(conf).all() and (conf >= min_confidence).all())
                if not ok:
                    values.append(np.nan)
                elif statistic == "value":
                    values.append(float(window[0]))
                else:
                    values.append(float(REDUCERS[statistic](window)))
            estimate = errors.lookup(trial.recording_mode, coordinate, planes.get(coordinate, "unspecified"), task)
            entry["sides"][side] = dict(status="computed", error_label=estimate.label, **_summary(values))
        left, right = entry["sides"]["left"]["mean"], entry["sides"]["right"]["mean"]
        if left is not None and right is not None:
            entry["left_minus_right"] = left - right
        output["parameters"][name] = entry
    return output


def _coordinate_for(variable, side, available):
    sided = f"{variable}_{side[0]}"
    if sided in available:
        return sided
    return variable if variable in available else None


def mean_curves(gait, variables):
    """{side: {variable: mean normalised curve or None}} from ``analyse_gait`` output."""
    result = {}
    for side in ("left", "right"):
        coords = gait["sides"][side]["coordinates"]
        result[side] = {}
        for variable in variables:
            coordinate = _coordinate_for(variable, side, coords)
            curves = [] if coordinate is None else [c for c in coords[coordinate]["normalized"] if c is not None]
            result[side][variable] = np.mean(curves, axis=0).tolist() if curves else None
    return result


def build_normative(gaits, *, variables, convention, source):
    """Reference mean and SD curves from several people's ``analyse_gait`` outputs.

    Each person contributes their per-side mean curve, so people with more cycles
    don't dominate. Build it from typically developing recordings processed with the
    *same* engine and conventions as the person being tracked.
    """
    if not gaits:
        raise ValueError("Need at least one gait analysis")
    if not variables or not convention.strip() or not source.strip():
        raise ValueError("variables, convention and source are required")
    curves = {v: [] for v in variables}
    for gait in gaits:
        for side_curves in mean_curves(gait, variables).values():
            for variable, curve in side_curves.items():
                if curve is not None:
                    curves[variable].append(curve)
    out = {}
    for variable, rows in curves.items():
        if not rows:
            raise ValueError(f"No complete cycles for {variable}")
        array = np.array(rows)
        out[variable] = dict(mean=array.mean(axis=0).tolist(),
                             sd=array.std(axis=0, ddof=1).tolist() if len(rows) > 1 else None,
                             n_curves=len(rows))
    samples = {len(v["mean"]) for v in out.values()}
    if len(samples) != 1:
        raise ValueError("Curves use different sample counts")
    return dict(schema_version=1, convention=convention, source=source, samples=samples.pop(),
                people=len(gaits), variables=out)


def save_normative(normative, path):
    Path(path).write_text(json.dumps(normative, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def load_normative(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    fields = {"schema_version", "convention", "source", "samples", "people", "variables"}
    if not isinstance(data, dict) or set(data) != fields or data["schema_version"] != 1:
        raise ValueError("Unsupported normative file")
    for variable, entry in data["variables"].items():
        if len(entry.get("mean", [])) != data["samples"]:
            raise ValueError(f"Normative curve {variable} has the wrong length")
    return data


def gait_profile_score(gait, normative, *, convention):
    """GPS-style scores: per-variable RMS deviation and per-side/overall RMS of those.

    ``convention`` must equal the normative file's convention; mixing models (e.g.
    OpenSim vs Plug-in Gait vs HBM2) would score convention differences as pathology.
    """
    if convention != normative["convention"]:
        raise ValueError(f"Convention mismatch: trial '{convention}' vs normative '{normative['convention']}'")
    variables = list(normative["variables"])
    curves = mean_curves(gait, variables)
    result = dict(caveat=GPS_CAVEAT, convention=convention, normative_source=normative["source"],
                  normative_people=normative["people"], variables=variables, sides={})
    all_scores = []
    for side in ("left", "right"):
        scores = {}
        for variable in variables:
            curve = curves[side][variable]
            if curve is None:
                scores[variable] = None
                continue
            reference = np.asarray(normative["variables"][variable]["mean"])
            if len(curve) != len(reference):
                raise ValueError("Gait curves and normative curves use different sample counts")
            scores[variable] = float(np.sqrt(np.mean((np.asarray(curve) - reference) ** 2)))
        complete = [s for s in scores.values() if s is not None]
        all_scores.extend(complete)
        result["sides"][side] = dict(variable_scores=scores,
                                     score=float(np.sqrt(np.mean(np.square(complete)))) if len(complete) == len(variables) else None,
                                     missing=[v for v, s in scores.items() if s is None])
    complete_sides = all(result["sides"][s]["score"] is not None for s in ("left", "right"))
    result["overall"] = float(np.sqrt(np.mean(np.square(all_scores)))) if complete_sides else None
    return result
