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
