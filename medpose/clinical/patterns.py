"""Config-driven discussion prompts; never diagnosis or severity scoring."""
import json
from pathlib import Path

import numpy as np

from medpose.errors import ErrorTable, reliability, PROVISIONAL
from medpose.trial import _unique_keys

RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "patterns.json"
PATTERNS = {"crouch", "stiff knee", "equinus or toe walking", "scissoring",
            "trunk sway", "pelvic obliquity", "asymmetry"}


def load_rules(path=RULES_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, ValueError) as exc:
        raise ValueError("Could not read pattern rules") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "rules"} or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported pattern rules schema")
    validate_rules(data["rules"])
    return data["rules"]


def validate_rules(rules):
    if not isinstance(rules, list):
        raise ValueError("rules must be a list")
    fields = {"name", "coordinates", "plane", "task", "statistic", "operator", "threshold_deg", "status", "origin"}
    seen = set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != fields or rule["name"] not in PATTERNS or rule["name"] in seen:
            raise ValueError("Malformed or duplicate pattern rule")
        seen.add(rule["name"])
        if rule["statistic"] not in {"minimum", "maximum", "range", "mean", "range_difference"} or rule["operator"] not in {"gt", "lt"}:
            raise ValueError("Unknown rule statistic or operator")
        coords = rule["coordinates"]
        count = 2 if rule["statistic"] == "range_difference" else 1
        if not isinstance(coords, list) or len(coords) != count or any(not isinstance(n, str) or not n for n in coords):
            raise ValueError("Invalid rule coordinates")
        for key in ("plane", "task", "origin"):
            if not isinstance(rule[key], str) or not rule[key].strip():
                raise ValueError(f"Rule {key} must be nonempty text")
        if rule["status"] != PROVISIONAL:
            raise ValueError("These candidate rules remain provisional: needs clinician review")
        threshold = rule["threshold_deg"]
        if threshold is not None and (type(threshold) not in (float, int) or not np.isfinite(threshold)):
            raise ValueError("Threshold must be finite degrees or null (disabled)")


def pattern_prompts(trial, *, errors=None, rules=None):
    """Whole-clip screening candidates only; no phase-specific diagnosis.

    All shipped thresholds are null. Enabling a rule requires a reviewer-chosen
    provisional threshold. Missing coordinates/data produce unavailable results.
    Confidence is reported without inventing a low-confidence cutoff.
    """
    rules = load_rules() if rules is None else rules
    validate_rules(rules)
    errors = ErrorTable.load() if errors is None else errors
    results = []
    for rule in rules:
        result = dict(name=rule["name"], status="disabled: no threshold", prompt=None,
                      threshold_status=rule["status"], threshold_origin=rule["origin"],
                      error_labels=[], reliability=[], scope="whole clip; not phase-specific")
        if rule["threshold_deg"] is None:
            results.append(result)
            continue
        coords = rule["coordinates"]
        if not len(trial.time) or any(n not in trial.coordinate_names for n in coords):
            result["status"] = "unavailable: coordinate or observations missing"
            results.append(result)
            continue
        indices = [trial.coordinate_names.index(n) for n in coords]
        if any(trial.units[i] != "deg" for i in indices):
            raise ValueError("Pattern rules require angular coordinates in degrees")
        values = trial.values[:, indices]
        if not np.isfinite(values).all():
            result["status"] = "unavailable: missing observations"
            results.append(result)
            continue
        if rule["statistic"] == "range_difference":
            value = abs(float(np.ptp(values[:, 0]) - np.ptp(values[:, 1])))
        else:
            reducer = {"minimum": np.min, "maximum": np.max, "mean": np.mean, "range": np.ptp}[rule["statistic"]]
            value = float(reducer(values))
        for coord in coords:
            error = errors.lookup(trial.recording_mode, coord, rule["plane"], rule["task"])
            result["error_labels"].append(error.label)
            result["reliability"].append(reliability(error).status)
        result["value_deg"] = value
        result["threshold_deg"] = rule["threshold_deg"]
        fired = value > rule["threshold_deg"] if rule["operator"] == "gt" else value < rule["threshold_deg"]
        result["status"] = "candidate prompt" if fired else "configured rule not triggered"
        if fired:
            text = f"{rule['name'].capitalize()}: worth discussing with a clinician. Provisional whole-clip rule; not a diagnosis."
            if any(s != "reliable" for s in result["reliability"]):
                text += " Based on unreliable or unmeasured values: " + "; ".join(result["error_labels"]) + "."
            else:
                text += " Benchmark error only; this person's true error remains unknown."
            if trial.confidence is None:
                text += " Confidence was not supplied."
            else:
                confidence = trial.confidence[:, indices]
                text += " Coordinate confidence minimum: " + (str(float(np.nanmin(confidence))) if np.isfinite(confidence).any() else "unknown") + "."
            if any("ankle" in n for n in coords):
                text += " The ankle reading reflects foot and brace together."
            result["prompt"] = text
        results.append(result)
    return results
