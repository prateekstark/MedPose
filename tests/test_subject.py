import pytest
import yaml

from medpose.subject import SubjectError, load_core_session, load_mono_subject


def write(tmp_path, **fields):
    path = tmp_path / "subject.yaml"
    path.write_text(yaml.safe_dump(fields))
    return path


def test_child_values_are_accepted(tmp_path):
    subject = load_mono_subject(write(tmp_path, height_m=1.3, mass_kg=28.5, sex="male"))
    assert (subject.height_m, subject.mass_kg, subject.sex) == (1.3, 28.5, "male")


@pytest.mark.parametrize("missing", ["height_m", "mass_kg", "sex"])
def test_mono_has_no_silent_defaults(tmp_path, missing):
    fields = {"height_m": 1.3, "mass_kg": 28.5, "sex": "female"}
    del fields[missing]
    with pytest.raises(SubjectError, match=missing):
        load_mono_subject(write(tmp_path, **fields))


@pytest.mark.parametrize("fields, message", [
    ({"height_m": 130, "mass_kg": 28.5}, "units"),      # centimetres
    ({"height_m": 1.3, "mass_kg": 1000}, "units"),
    ({"height_m": "tall", "mass_kg": 28.5}, "number"),
    ({"height_m": None, "mass_kg": 28.5}, "missing"),
    ({"height_m": True, "mass_kg": 28.5}, "number"),
])
def test_bad_numbers_are_refused(tmp_path, fields, message):
    with pytest.raises(SubjectError, match=message):
        load_core_session(write(tmp_path, **fields))


def test_core_allows_blank_sex_like_upstream_sessions(tmp_path):
    assert load_core_session(write(tmp_path, height_m=1.82, mass_kg=100.0, gender_mf=None)).sex is None


def test_template_values_trigger_a_warning(tmp_path, capsys):
    load_mono_subject(write(tmp_path, height_m=1.7, mass_kg=70.0, sex="male"))
    assert "template values" in capsys.readouterr().out
    load_mono_subject(write(tmp_path, height_m=1.3, mass_kg=28.5, sex="male"))
    assert "template values" not in capsys.readouterr().out


def test_unreadable_or_empty_file_is_refused(tmp_path):
    with pytest.raises(SubjectError):
        load_mono_subject(tmp_path / "missing.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(SubjectError):
        load_mono_subject(empty)
