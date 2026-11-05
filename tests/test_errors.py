"""All numeric errors here are simulated test inputs, not accuracy claims."""
import json

import numpy as np
import pytest

from medpose.errors import ErrorEstimate, ErrorTable, ReliabilityConfig, PROVISIONAL, reliability


def entry(**changes):
    data = dict(mode="1_phone", coordinate="knee_angle_l", plane="sagittal", task="walking",
                value=3., source="simulated", origin="synthetic test fixture", unit="deg")
    data.update(changes)
    return data


def test_empty_shipping_table_and_exact_lookup():
    missing = ErrorTable.load().lookup("1_phone", "knee_angle_l", "sagittal", "walking")
    assert missing.value is None
    assert missing.label == "error not yet measured"
    assert reliability(missing).status == "unknown"
    table = ErrorTable([entry()])
    found = table.lookup("1_phone", "knee_angle_l", "sagittal", "walking")
    assert found.value == 3
    assert "simulated" in found.label and "synthetic test fixture" in found.label
    assert reliability(found).status == "unknown"
    for key in [("2_phone", "knee_angle_l", "sagittal", "walking"),
                ("1_phone", "knee_angle_r", "sagittal", "walking"),
                ("1_phone", "knee_angle_l", "frontal", "walking"),
                ("1_phone", "knee_angle_l", "sagittal", "squat")]:
        assert table.lookup(*key).value is None


@pytest.mark.parametrize("value,source,expected", [
    (5., "measured", "reliable"), (5.01, "measured", "not reliable"),
    (0., "measured", "reliable"), (3., "simulated", "unknown"),
    (3., "provisional", "unknown"), (6., "simulated", "not reliable"),
])
def test_threshold_boundary(value, source, expected):
    # Synthetic exercise of the source routing: <=5 passes, >5 fails.
    # 'measured' is a test label only, not an observation on people.
    result = reliability(ErrorEstimate(value, source, "synthetic routing test", "deg"))
    assert result.status == expected
    assert result.threshold_status == PROVISIONAL
    assert "not been verified" in result.threshold_origin


def test_metres_never_use_degree_limit():
    error = ErrorEstimate(.1, "measured", "synthetic routing test", "m")
    assert reliability(error).status == "unknown"
    config = ReliabilityConfig(5., .05, PROVISIONAL, "synthetic threshold, not a clinical claim")
    assert reliability(error, config).status == "not reliable"


@pytest.mark.parametrize("change", [{"value": -1}, {"value": np.nan}, {"value": np.inf},
    {"value": None}, {"value": True}, {"source": "published"}, {"origin": ""},
    {"unit": "m"}, {"mode": "bad"}, {"task": ""}, {"extra": 1}])
def test_malformed_entries(change):
    with pytest.raises(ValueError):
        ErrorTable([entry(**change)])


def test_duplicate_keys():
    with pytest.raises(ValueError, match="Duplicate"):
        ErrorTable([entry(), entry()])


@pytest.mark.parametrize("text", ["", "null", "{}", '{"schema_version":2,"entries":[]}',
    '{"schema_version":true,"entries":[]}', '{"schema_version":1,"entries":null}',
    '{"schema_version":1,"schema_version":1,"entries":[]}'])
def test_malformed_files(tmp_path, text):
    path = tmp_path / "synthetic_errors.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        ErrorTable.load(path)


def test_file_roundtrip(tmp_path):
    path = tmp_path / "synthetic_errors.json"
    path.write_text(json.dumps(dict(schema_version=1, entries=[entry()])))
    assert ErrorTable.load(path).lookup("1_phone", "knee_angle_l", "sagittal", "walking").source == "simulated"


@pytest.mark.parametrize("limit", [-1, np.inf, np.nan, True])
def test_invalid_threshold(limit):
    with pytest.raises(ValueError):
        ReliabilityConfig(limit, None, PROVISIONAL, "synthetic")


def test_missing_origin_and_invalid_config_file(tmp_path):
    with pytest.raises(ValueError):
        ErrorEstimate(1, None, None, "deg")
    with pytest.raises(ValueError):
        ReliabilityConfig(5, None, "cited", "unverified attribution")
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        ReliabilityConfig.load(path)
