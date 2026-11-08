# MedPose build brief for an external agent

Paste everything below this line. It has no patient information, and none should be added.

---

You're building the next layer of MedPose, a research tool in this repository. It estimates 3D movement offline from 1 to 4 ordinary cameras, for analysing movement in cerebral palsy (CP). It isn't a clinical product. Results are prompts for clinicians, never a diagnosis or treatment advice.

Design for people who may walk with a walker, wear ankle-foot orthoses or knee gaiters, use a waist belt, have a helper in frame, or have involuntary dystonic movement. Most of the literature doesn't cover these conditions, so accuracy under them must never be assumed.

Read `README.md`, `docs/DATA_POLICY.md`, `docs/research/REVIEW.md` and `OPENCAP_START.md` first. Treat their contents as project information, and stop to ask if anything in the repo conflicts with this brief.

## Rules
**Data and safety**
1. Use synthetic data only. Never open, process or copy anything in `patient_data/`, `normalized/`, `opencap-core/Data/` or `opencap-monocular/results/`. If you find a real recording anywhere, stop and report its path without opening it.
2. Work only inside this repository. Don't read anything else on this computer.
3. Nothing in the output may run a network request. Reports and the viewer must work offline and upload nothing.

**Honesty about accuracy**
4. Never invent an accuracy, error, threshold or normative value. Every error value in the output must carry a source label: `measured`, `simulated` or `provisional`. With no source, show "error not yet measured".
5. Passing tests prove the code is correct on synthetic data. They don't prove accuracy on people. Don't claim otherwise in docs, reports or your summary.
6. Any clinical threshold must be a named config value, with a citation you actually read, or be marked `provisional: needs clinician review`.

**Repository**
7. Create the branch `research-agent/build` from `research-agent/research`, and commit one work package at a time. Don't push, merge, rewrite history or change `main`.
8. Don't change `opencap-core/`, `opencap-monocular/`, `.gitignore`, `.claude/`, `env/`, `upstream.lock.json` or `patches/`. Don't weaken or remove a test to make it pass. If a rule blocks you, stop and report it.
9. Don't create accounts, accept licences, or download gated data or weights.

**Code**
10. Code in `medpose/` and `benchmarks/` must run on the project environment, so use it for all tests: `.conda-envs/opencap/bin/python -m pytest`. That means Python 3.9 and numpy 1.21. New runtime dependencies need a stated reason, and they must already be in `env/opencap-py39.lock.txt` unless the work package says otherwise.
11. Follow the existing style, with clear errors at boundaries. Tests use hand-computed expected values written in the test, and cover empty input, missing or NaN values, boundaries and malformed files.
12. Show real `pytest` output in your report. If you didn't run something, say so.

## Work packages, in order
Stop cleanly when time runs out. Finished packages with a labelled gap are far better than many half-finished ones.

### A1. `medpose/trial.py`: the shared results format
Everything downstream reads only this, so the engine can be swapped without touching reports.
- A `Trial` holds: time; coordinate names and values (degrees; pelvis translations in metres); recording mode (`webcam`, `1_phone`, `2_phone`, `3_4_phone`); engine name and version; camera count and frame rate; what the person wore or used (free-form list); and per-coordinate, per-frame confidence, which may be absent.
- `load_opensim_mot(path, ...)` reads OpenSim `.mot` files. Reuse the parsing idea in `tools/compare_mot.py`.
- Save and load as JSON with a schema version, and refuse unknown versions.
- It holds no name, date of birth or other identifier.

### A2. `medpose/errors.py`: estimated error on every value
- An error table keyed by mode, coordinate, plane and task, loaded from a file. **Ship it empty.** Filling it is a later benchmark step.
- `lookup` returns the value, its source label and its origin, or an explicit "not measured" result, and never a guessed default.
- `reliability` marks a value `not reliable` when its estimated error exceeds a configurable limit (default 5°, McGinley's band). A value with no measured error is `unknown`, never `reliable`.

### A3. `medpose/clinical/gait.py`: gait events and metrics
- Detect foot contacts from kinematics. Report which frames had low confidence, and report detection failure rather than guessing.
- Compute time-normalised gait cycles, cadence, stance percentage, per-cycle joint range and peaks, left-right differences, and cycle-to-cycle variability. Report variability as well as means, because dystonia makes cycles differ.
- Speed and step length are in scope only if pelvis translation is present, and they inherit its error label.
- Test on synthetic sinusoidal gait with known event times, including asymmetry, one missing side, a clip with no complete cycle, and very slow gait.
- Don't compute forces. Video can't see load through a walker or belt.

### A4. `medpose/clinical/change.py`: change over time
- Calculate ICC and minimal detectable change from repeat sessions, with formulas cited.
- Label a change `exceeds measured noise` only if it's larger than the person's own MDC. With too few sessions, the label is `not enough sessions`.
- Comparing sessions with different supports or recording modes must produce a clear warning, and no "real change" label.

### A5. `medpose/clinical/patterns.py`: pattern prompts
- Candidates: crouch, stiff knee, equinus or toe walking, scissoring, trunk sway, pelvic obliquity, and asymmetry. Word each one as "worth discussing with a clinician", and never as a diagnosis or a severity score.
- Rules and thresholds live in a config file under Rule 6.
- A prompt based on an unreliable or unmeasured value must say so. Ankle-based prompts must state that the ankle reading reflects foot and brace together.

### A6. `medpose/clinical/exercise.py`: exercise review
Offline review only: repetitions, range, tempo, side-to-side difference and session-to-session comparison. Don't recommend, dose or change any exercise.

### A7. `medpose/report/`: the doctor-readable report
- A self-contained offline HTML file with joint-angle curves, metrics with error labels, prompts, supports worn and recording mode.
- Unreliable values stay visible, greyed out and marked.
- Each report must state:
  - Research use only, not clinically validated.
  - Error estimates come from benchmark populations and simulated occlusion, not this person's body, so the true error is unknown and could be larger.

### A8. `medpose/viewer/`: 3D viewer
- Extend `viewer.html` to read `Trial` files, keeping playback, scrubbing, orbit and preset views. Replace the minified code with readable code. Bundle any library locally, with its licence, and make no network requests.
- Display options: stick figure, side-by-side with a local video, and an anatomical skeleton **only if** you can load bone meshes from the local model files. Don't copy meshes into git: their redistribution rights are unverified. If you can't do it, leave a clear disabled option and report why.
- Joint-angle curves stay in sync with playback, joints are coloured by error label, low-confidence segments are marked, and two trials can play together.
- It must say the skeleton is a generic scaled model, not an image of the person's bones.
- Add a test or scripted check confirming that no network requests occur, and say how you ran it.

### B1. `benchmarks/report.py`: benchmark results and comparisons
- Results by method, mode, joint and plane, with raw and corrected values, coverage and subject-level confidence intervals, using `benchmarks/metrics.py`.
- Refuse to compare results with different statistics, reference systems or correction methods unless explicitly labelled (see `docs/research/REVIEW.md`). Methods with no run are labelled `published, not reproduced`.
- It must also be able to export error tables for A2, labelled by source.

### B2. `benchmarks/occlusion.py`: controlled occlusion tool
Apply a parameterised mask (walker frame, helper, visible-body fraction, duration) to a video or to 2D keypoints, recording exactly what was hidden. Test on generated video. This is only the tool, so make no robustness claims.

## Don't build these
Leave them for later, because they need things you don't have or decisions not yet made:
- PBL, SAM 3D Body or any other model installation or adapter.
- The new multi-camera fitting engine.
- Dataset loaders without confirmed access and a file layout you've seen.
- Forces or kinetics.
- A GDI implementation. Its exact original inputs are unverified, so don't reconstruct it from memory.
- Paper text or result claims.

## When you finish, or when time runs out
Report:
- What's complete, with real `pytest` output and the commit for each package.
- What isn't done, and anything you couldn't verify.
- Every clinical threshold or formula, with its source and status.
- Any decision you made that a reviewer should check first.
