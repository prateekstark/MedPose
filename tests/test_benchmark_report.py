import numpy as np
import pytest

from benchmarks.report import benchmark_results, compare_results, export_error_table, published_result
from medpose.errors import ErrorTable
from medpose.trial import Trial


def trial(values):
    return Trial(np.arange(len(values), dtype=float), ("knee",), np.asarray(values).reshape(-1, 1),
                 "1_phone", "synthetic", "1", 1, 1)


def report(data):
    return benchmark_results(data, method="synthetic", planes={"knee":"sagittal"}, task="walking",
        reference_system="synthetic truth", source="simulated", origin="generated fixture", n_boot=100, seed=0)


def test_raw_corrected_coverage_and_export(tmp_path):
    # Subject A error [1,3] -> MAE2, corrected1. B [5,5] -> MAE5, corrected0.
    # Equal-subject means: raw3.5; corrected .5. One missing of five total ->80%.
    rows = report({"a":[(trial([1,3,np.nan]),trial([0,0,0]))], "b":[(trial([5,5]),trial([0,0]))]})
    assert rows[0]["value"] == 3.5 and rows[1]["value"] == .5
    assert rows[0]["coverage"] == .8
    assert rows[0]["source"] == "simulated" and rows[0]["ci95"] is not None
    path = tmp_path / "errors.json"
    export_error_table(rows[:1], path)
    estimate = ErrorTable.load(path).lookup("1_phone", "knee", "sagittal", "walking")
    assert estimate.value == 3.5 and estimate.source == "simulated"
    with pytest.raises(ValueError):
        export_error_table(rows, path)


def test_comparison_guards_and_published_status():
    row = report({"a":[(trial([1,3]),trial([0,0]))]})[0]
    assert row["ci95"] is None
    for key in ("statistic", "reference_system", "correction", "aggregation"):
        other = dict(row); other[key] = "different"
        with pytest.raises(ValueError):
            compare_results([row, other])
        assert compare_results([row, other], mismatch_label="different protocols; descriptive only")["warning"]
    published = published_result(**{key: row[key] for key in ("method","mode","coordinate","plane","task","unit","statistic","aggregation","reference_system","correction","value","source","origin")})
    assert published["run_status"] == "published" and published["coverage"] is None


def test_empty_missing_and_mismatch():
    assert report({}) == []
    rows = report({"a":[(trial([np.nan]),trial([0]))]})
    assert rows[0]["value"] is None and rows[0]["subjects_excluded"] == 1
    with pytest.raises(ValueError):
        report({"a":[(trial([1,2]),trial([0]))]})
    with pytest.raises(ValueError):
        compare_results([{}])
