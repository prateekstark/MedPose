# Build packages: synthetic validation only

Branch: `research-agent/build`, based on `research-agent/research` at `2dddf0b`.
No accuracy on people is established by these tests. No new runtime dependencies.

## A1 — Trial format

Implemented `medpose/trial.py` and synthetic tests. Schema version 1 rejects unknown
versions, extra fields and duplicate JSON keys. No identifier fields or source paths
are stored. Free-text equipment/engine fields must not contain personal details;
the schema cannot detect identifying text entered into those fields.

Angles use degrees; only pelvis_tx/ty/tz use metres. NaN is JSON null. Time must be
finite and increasing; acquisition frame rate does not override the timestamps.
Confidence is optional, coordinate-specific, in [0,1] or missing; that interval is
a representation constraint, not a clinical threshold. Empty Trials are supported;
empty MOT files fail. MOT loading requires declared angular units (or an explicit
caller argument), and explicit confirmation of metre units for translations.
Radians are converted to degrees. Header identifiers are discarded.

Command:
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_trial.py --basetemp=.pytest_cache/trial
```
Actual output:
```text
platform linux -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
collected 42 items
tests/test_trial.py ..........................................           [100%]
============================== 42 passed in 0.18s ==============================
```
The initial unmodified pytest command failed during unrelated ROS plugin loading
(`ModuleNotFoundError: osrf_pycommon`). Automatic plugin loading is disabled for
these runs; no project tests or environment packages were changed. An intermediate
run also failed to create a nested temporary directory; subsequent runs use the
existing repository-local pytest cache so generated fixtures stay in the repo.

Review first: schema versioning, explicit unit confirmation, free-text privacy,
and the A3 input contract. Joint angles alone do not specify foot positions or the
ground plane. Contact detection must not be fabricated from angle peaks.

A1 commit: `99cb985`.

## A2 — Error estimates and reliability

Implemented `medpose/errors.py`, an empty `medpose/config/errors.json`, and named
limits in `medpose/config/reliability.json`. Lookup requires an exact mode,
coordinate, plane and task match. Every numeric estimate requires measured,
simulated or provisional source plus an origin and compatible unit. No entry
means **error not yet measured**, never a fallback value. Duplicate rows, unknown
schema versions, malformed rows and unit mismatches fail explicitly.

Threshold inventory:

| Config | Value | Source/status |
|---|---|---|
| angular_limit_deg | 5 degrees | User-requested default; **provisional: needs clinician review**. McGinley attribution was not verified and is not presented as a citation. |
| translation_limit_m | absent | No translation cutoff invented; reliability remains unknown. |

A measured error equal to the limit passes the configured check; strictly greater
fails. Simulated/provisional estimates below the limit remain unknown. Every
result preserves threshold status and origin. “Reliable” means only that the
measured benchmark estimate meets the configured limit, not validation on this
person. A provisional threshold remains provisional even when an estimate passes.

Command:
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_trial.py tests/test_errors.py --basetemp=.pytest_cache/a2
```
Actual output:
```text
platform linux -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
collected 75 items
tests/test_trial.py ..........................................           [ 56%]
tests/test_errors.py .................................                   [100%]
============================== 75 passed in 0.29s ==============================
```

No clinical formulas implemented in A1/A2. The degrees/radians conversion is a unit
conversion. Tests exercise synthetic inputs, including simulated routing of the
`measured` label; those tests are not human error measurements. No full-project
regression run or browser/network-request check has been performed in these packages.

## Earlier pause (resolved)

A3–A8 and B1–B2 are not implemented. A3 needs an input decision: the requested Trial
contains angles and pelvis translations, not foot positions or a ground plane.
The user has been asked whether to extend Trial with optional foot positions and
a ground-plane convention or explicitly report contact detection unavailable.
No contacts will be guessed from angle peaks. The user subsequently approved optional foot positions; see A3 below.

Review first: the contact-input decision; whether the provisional angular limit
is suitable; the interpretation of reliable as benchmark-only; free-text fields
cannot guarantee absence of identifiers. The schema has no personal identifier
fields. Later reports must retain the caveats and provenance carried here.

## A3 — Foot positions, contact candidates and gait summaries

The user approved optional foot positions. Trial schema 2 adds world-frame heel/toe
**sole contact locations**, a unit upward ground normal, plane offset in metres,
optional per-point confidence, and non-identifying provenance. Schema 1 still
loads without feet; unknown versions fail. Raw skin-marker positions must be
mapped to sole locations by the engine; this package does not invent that mapping.
Foot and pelvis positions must share the same world frame.

Implemented provisional ground-proximity event candidates, complete cycles,
time-normalized curves, cadence, geometric stance percentage, range/peaks,
left-minus-right mean range differences and per-cycle values/mean/sample SD.
No complete cycle is reported across missing positions, low confidence or a time
gap exceeding the supplied limit. A single cycle has no estimated sample SD.
Missing coordinate observations invalidate that coordinate's cycle; no gap is
filled. Absent foot confidence is explicitly reported, never invented.

All six detector settings must be supplied: clearance_m,
penetration_tolerance_m, min_confidence, max_gap_s, min_stance_s, min_swing_s.
All are **provisional: needs clinician review**, with no shipped clinical values.
The numbers in tests are simulated fixture choices, not normative cutoffs.
The detector cannot establish load/contact truth from geometric proximity.
Cadence = 120 / ipsilateral cycle seconds assumes alternating steps; geometric
stance percent = 100 * proximity seconds / cycle seconds. These are descriptive
computations with provisional event timing, not validated clinical formulas.

An explicit world-frame progression direction enables signed pelvis displacement
per cycle / duration (speed) and leading-minus-trailing heel projection at candidate
onset (step length). Both require pelvis translations. Source pelvis error labels
are retained, but no numeric speed/step error is fabricated by reusing a positional
error in different units; propagated errors remain **error not yet measured**.
No forces are computed. No human accuracy is inferred from this implementation.

Command:
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_trial.py tests/test_errors.py tests/test_gait.py --basetemp=.pytest_cache/a3
```
Actual output:
```text
collected 86 items
tests/test_trial.py ..........................................           [ 48%]
tests/test_errors.py .................................                   [ 87%]
tests/test_gait.py ...........                                           [100%]
============================== 86 passed in 0.42s ==============================
```
An initial variability fixture changed the endpoint shared by two cycles, so its
hand calculation was wrong. It now uses a waveform with zero-valued boundaries;
assertions still require the hand-computed mean 30 and sample SD sqrt(200).

Review first: sole-point mapping and shared world frame; all detector settings;
provisional cadence assumptions; event timing quantization to the first sampled
frame; absent confidence; and lack of validated uncertainty propagation.

A3 commit: `787d0de`.

## A4 — Repeatability and change

`icc_absolute_agreement` implements balanced subjects-by-repeat-session ICC(A,1),
with complete-case subject exclusions reported. It is not a person's own ICC.
Formula: (MS_subject - MS_error) / (MS_subject + (k-1)*MS_error +
(k/n)*(MS_session-MS_error)). [Liljequist et al. 2019, Eq.12](https://doi.org/10.1371/journal.pone.0219854)
was read in the publisher PDF; source verified. No quality bands are imposed.

Personal MDC95 = 1.96 * sqrt(2) * sample SD of this person's stable, independent
repeat-session scalar measurements. The repeatability formula was read on
[Martin Bland's author page](https://www-users.york.ac.uk/~mb55/meas/accept.htm).
Its application assumes stable normally distributed independent measurement noise;
biological variability is included. It is not an estimate from a cohort ICC.
A minimum repeat-session count is caller-configured, >=2, **provisional: needs
clinician review**; two is only the mathematical minimum. No clinical sample-size
adequacy claim is made. The 1.96 constant is the cited normal 95% multiplier.

Only a measured-source, nondegenerate personal estimate and a strictly larger
absolute change permit `exceeds measured noise`. Context differences (including
supports, mode and engine) suppress the label. Missing/too-few sessions and
unconfirmed stable baseline have explicit statuses. Reusing the same Trial object
as independent sessions is rejected. Caller must verify person identity, metric
and task equivalence, independence and baseline stability; identifiers are not
stored, so software cannot establish those facts.

Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_change.py --basetemp=.pytest_cache/a4`
```text
collected 7 items
tests/test_change.py .......                                             [100%]
============================== 7 passed in 0.06s ===============================
```

A4 commit: `6a50b8b`.

## A5 — Provisional pattern prompts

Seven requested candidates are configured in `medpose/config/patterns.json`.
Every threshold ships null (disabled). No normative angle is invented. A reviewer
must explicitly choose a provisional threshold, coordinate sign/side, statistic
and task interpretation. Enabled rules are whole-clip descriptive screens, not
phase-specific diagnostic criteria. Missing data produces unavailable output.
Prompts say “worth discussing with a clinician”, disclose unavailable/unreliable
errors, and ankle prompts state that the reading reflects foot and brace together.
No diagnosis, severity score or treatment recommendation is generated.

All rule thresholds: **provisional: needs clinician review**; no clinical cutoff
or clinical formula is cited or claimed. Range=max-min; asymmetry uses absolute
difference between left/right ranges. These are descriptive summaries only.

Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_patterns.py --basetemp=.pytest_cache/a5`
```text
collected 4 items
tests/test_patterns.py ....                                              [100%]
============================== 4 passed in 0.05s ===============================
```

A5 commit: `6e381a4`.

## A6 — Offline exercise review

Complete low-high-low excursions produce repetition count, per-repetition range,
total/rising/returning duration, side differences and descriptive session
comparisons. Incomplete excursions, NaNs, low confidence and excessive time gaps
are not bridged. No recommendation, dose or exercise change is produced. Context
mismatches suppress session differences; otherwise no real-change label is given
without A4 personal repeatability analysis.

low_deg, high_deg, min_confidence and max_gap_s are explicit caller settings,
all **provisional: needs clinician review**. There are no shipped clinical values.
Tempo is elapsed timestamp difference, range is max-min; neither is a clinical
threshold. Side comparisons are differences of mean ranges, not matched reps.

Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_exercise.py --basetemp=.pytest_cache/a6`
```text
collected 7 items
tests/test_exercise.py .......                                           [100%]
============================== 7 passed in 0.04s ===============================
```

A6 commit: `edb3c60`.

## A7 — Offline HTML report

`render_report` / `save_report` produce standalone HTML/SVG with coordinate curves,
missing-frame counts, error labels, benchmark reliability, supports, recording
mode, optional gait summaries and A5 prompts. Unknown/unreliable data remain
visible in grey. Required research and individual-error caveats are included.
Text is HTML-escaped; gaps are not connected. The output contains no scripts,
external assets or outbound URLs; a restrictive content security policy is added.
No dependencies added. Full browser visual QA remains a gap; automated tests
inspect generated structure, escaping, missing points and prohibited resource tags.

Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_report.py --basetemp=.pytest_cache/a7`
```text
collected 2 items
tests/test_report.py ..                                                  [100%]
============================== 2 passed in 0.07s ===============================
```
