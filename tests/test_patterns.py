import json

import numpy as np
import pytest

from medpose.trial import Trial
from medpose.clinical.patterns import load_rules, pattern_prompts


def trial(values=((0,), (10,))):
    return Trial(np.arange(len(values), dtype=float), ("ankle_angle_l",), np.asarray(values).reshape(-1, 1),
                 "1_phone", "synthetic", "1", 1, 1)


def test_shipped_rules_disabled():
    results = pattern_prompts(trial())
    assert len(results) == 7
    assert all(r["prompt"] is None and r["status"].startswith("disabled") for r in results)


def rule():
    result = load_rules()[2]
    result.update(statistic="range", operator="gt", threshold_deg=5.)
    return result


def test_provisional_prompt_and_ankle_caveat():
    # Synthetic range max10-min0=10 > fixture threshold5.
    result = pattern_prompts(trial(), rules=[rule()])[0]
    assert result["value_deg"] == 10
    assert "worth discussing with a clinician" in result["prompt"]
    assert "error not yet measured" in result["prompt"]
    assert "foot and brace together" in result["prompt"]
    assert "not a diagnosis" in result["prompt"]
    assert pattern_prompts(trial(((0,), (5,))), rules=[rule()])[0]["prompt"] is None


def test_empty_missing_and_nan():
    for values in ([], [[np.nan]]):
        assert pattern_prompts(trial(values), rules=[rule()])[0]["status"].startswith("unavailable")
    missing = rule(); missing["coordinates"] = ["missing"]
    assert pattern_prompts(trial(), rules=[missing])[0]["prompt"] is None


def test_malformed_rules(tmp_path):
    path = tmp_path / "rules.json"
    for text in ("", "{}", json.dumps({"schema_version":1, "rules":[{}]})):
        path.write_text(text)
        with pytest.raises(ValueError):
            load_rules(path)
    invalid = rule(); invalid["threshold_deg"] = np.nan
    with pytest.raises(ValueError):
        pattern_prompts(trial(), rules=[invalid])
