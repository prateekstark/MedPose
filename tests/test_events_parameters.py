"""Zeni events, gait parameters, GPS-style score, report and CLIs on generated walking."""
import json

import numpy as np
import pytest

from gait_synthetic import expected_left_strikes, walking_trial
from medpose.analyse import analyse_walk, load_event_settings
from medpose.clinical.events import event_times, progression_direction, zeni_events
from medpose.clinical.parameters import build_normative, gait_profile_score
from medpose.report.gait_report import render_gait_report
from medpose.trial import Trial

SETTINGS = dict(load_event_settings(), min_step_interval_s=0.4, max_gap_s=0.1)


def test_direction_follows_the_pelvis_and_refuses_turns():
    trial = walking_trial()
    direction, details = progression_direction(trial, min_distance_m=1, min_straightness=0.8)
    np.testing.assert_allclose(direction, [1, 0, 0], atol=1e-9)
    values = trial.values.copy()
    values[:, 0] = np.where(trial.time < 2.6, trial.time, 5.2 - trial.time)  # walks out and back
    turned = Trial(trial.time, trial.coordinate_names, values, "2_phone", "s", "1", 2, 100.0,
                   foot_positions=trial.foot_positions)
    with pytest.raises(ValueError, match="straightness|travelled"):
        progression_direction(turned, min_distance_m=1, min_straightness=0.8)


def test_zeni_finds_the_generated_events():
    trial = walking_trial()
    events = zeni_events(trial, "left", direction=[1, 0, 0], min_step_interval_s=0.4, max_gap_s=0.1)
    times = event_times(trial, events)
    assert times["foot_strike"] == pytest.approx(expected_left_strikes(), abs=0.011)
    assert times["foot_off"] == pytest.approx([0.6, 1.6, 2.6, 3.6, 4.6], abs=0.011)
    assert len(events["cycles"]) == 4
    assert "not validated in children with CP" in events["validation"]


def test_zeni_does_not_bridge_missing_feet():
    trial = walking_trial()
    left = trial.foot_positions.left.copy()
    left[250:260] = np.nan
    feet = type(trial.foot_positions)(left, trial.foot_positions.right, [0, 1, 0], 0.0, "generated")
    gapped = Trial(trial.time, trial.coordinate_names, trial.values, "2_phone", "s", "1", 2, 100.0,
                   foot_positions=feet)
    events = zeni_events(gapped, "left", direction=[1, 0, 0], min_step_interval_s=0.4, max_gap_s=0.1)
    for cycle in events["cycles"]:
        assert not (cycle["start"] <= 255 <= cycle["stop"])


def test_parameters_match_the_generated_knee_ramp():
    gait, params = analyse_walk(walking_trial(), settings=SETTINGS)
    assert gait["sides"]["left"]["stance_percent"]["mean"] == pytest.approx(60, abs=1.5)
    assert gait["speed"]["sides"]["left"]["mean"] == pytest.approx(1.0, abs=1e-6)
    p = params["parameters"]
    assert p["knee_flexion_at_initial_contact"]["sides"]["left"]["mean"] == pytest.approx(0, abs=0.7)
    assert p["knee_flexion_min_stance"]["sides"]["left"]["mean"] == pytest.approx(0, abs=0.7)
    assert p["knee_flexion_peak_swing"]["sides"]["left"]["mean"] == pytest.approx(59.4, abs=0.7)
    assert p["knee_flexion_peak_swing"]["sides"]["left"]["count"] == 4
    assert p["knee_flexion_peak_swing"]["sides"]["left"]["error_label"] == "error not yet measured"
    assert p["ankle_angle_at_initial_contact"]["left_minus_right"] is not None


def test_gps_style_score_is_zero_against_itself_and_reports_offsets():
    gait, _ = analyse_walk(walking_trial(), settings=SETTINGS)
    normative = build_normative([gait], variables=["knee_angle", "hip_flexion"], convention="c", source="generated")
    same = gait_profile_score(gait, normative, convention="c")
    assert same["sides"]["left"]["score"] == pytest.approx(0, abs=1e-9)
    shifted, _ = analyse_walk(walking_trial(knee_offset=5.0), settings=SETTINGS)
    scored = gait_profile_score(shifted, normative, convention="c")
    assert scored["sides"]["left"]["variable_scores"]["knee_angle"] == pytest.approx(5, abs=0.5)
    assert "not the published Gait Profile Score" in scored["caveat"]
    with pytest.raises(ValueError, match="Convention mismatch"):
        gait_profile_score(gait, normative, convention="Plug-in Gait")


def test_gait_report_is_offline_and_shows_parameters():
    trial = walking_trial()
    gait, params = analyse_walk(trial, settings=SETTINGS)
    normative = build_normative([gait, gait], variables=["knee_angle"], convention="c", source="generated")
    html = render_gait_report(trial, gait, params, task="walking", events_method="Zeni", normative=normative,
                              gps=gait_profile_score(gait, normative, convention="c"),
                              previous_gait=gait, previous_parameters=params)
    assert "Most knee bend during swing" in html and "posterior walker" in html
    assert "<script" not in html and "http" not in html.replace("http-equiv", "")
    assert "descriptive" in html and "low trust" in html


def test_analyse_cli_writes_numbers_and_report(cli, tmp_path):
    path = tmp_path / "trial.json"
    walking_trial().save_json(path)
    result = cli("medpose.analyse", "run", path, "--output-dir", tmp_path / "out")
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "out" / "gait.json").read_text())
    assert len(data["gait"]["sides"]["left"]["events"]["cycles"]) == 4
    assert (tmp_path / "out" / "report.html").exists()
    again = cli("medpose.analyse", "run", path, "--output-dir", tmp_path / "out")
    assert again.returncode != 0 and "Refusing" in again.stderr


def test_normative_cli_then_run_with_reference(cli, tmp_path):
    paths = []
    for i, offset in enumerate((-1.0, 1.0)):
        paths.append(tmp_path / f"td{i}.json")
        walking_trial(knee_offset=offset).save_json(paths[-1])
    ref = tmp_path / "ref.json"
    built = cli("medpose.analyse", "normative", *paths, "--source", "generated", "--output", ref,
                "--variables", "knee_angle", "hip_flexion")
    assert built.returncode == 0, built.stderr
    target = tmp_path / "trial.json"
    walking_trial(knee_offset=4.0).save_json(target)
    run = cli("medpose.analyse", "run", target, "--normative", ref, "--output-dir", tmp_path / "out")
    assert run.returncode == 0, run.stderr
    gps = json.loads((tmp_path / "out" / "gait.json").read_text())["gait_profile"]
    assert gps["sides"]["left"]["variable_scores"]["knee_angle"] == pytest.approx(4, abs=0.5)
