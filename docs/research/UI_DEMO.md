# Offline review UI and synthetic demonstration

Open `medpose/ui/demo.html` directly in a modern browser. It is self-contained:
HTML, styles, scripts, rendered MP4, Trial data and the demo's Python-generated
report are embedded. No web server, account, remote fonts, libraries or internet
connection is required. Source template: `medpose/ui/workspace.html`.

## What works

- Synchronized sample video, timeline scrubbing, playback speed and a rotatable
  schematic movement model with front, side and perspective views.
- Left/right hip, knee and ankle curves; a selector for other supplied coordinates.
  Missing observations remain gaps, and error estimates remain unknown.
- Local Trial JSON import, separate local-video attachment and explicit video offset.
  Importing a Trial clears the demo video so unrelated data is not silently paired.
- Two-Trial descriptive comparisons, with clear support/mode/engine/frame-rate
  mismatch warnings and no real-change label without personal repeat-session data.
- Local benchmark-results import, source labels, published/reproduced status,
  distinct raw/corrected rows and warnings about mismatched protocols.
- Non-identifying notes held only in page memory and downloadable standalone HTML
  reports with curves, visible unknown errors, context and research disclaimers.
- Responsive desktop/mobile layouts and explicit empty/malformed-file states.

## Demo provenance and limits

The 240-frame, 30-fps video is generated from invented periodic curves. A stylized
walker prop is drawn into the frames. Both the Trial and rendered video are
**simulated**, not observations of a person. The application did not estimate pose
from this video. The displayed ranges are descriptive values of those invented
curves, not clinical thresholds or normative values.

The stick figure uses arbitrary schematic proportions. It is an in-place hip/knee
illustration, not a full anatomical reconstruction or a person's bones. The ankle
curve remains available, but the simplified foot segment is not driven by it.
No meshes, weights or datasets were downloaded or opened.

No new clinical threshold or clinical formula was added. The browser does not
reimplement gait-event detection, exercise segmentation, personal MDC or clinical
pattern rules. Those remain in the tested Python modules and require explicit
inputs/settings. No contact metrics are invented for this demo. Benchmark values
are not automatically applied to a person's Trial.

Attaching a local video does not run inference, calibrate a camera, establish
accuracy or prove synchronization. The user must check provenance and alignment.
The complete Python pipeline and model adapters are not wired into this browser
workspace. Synthetic tests do not establish performance on people.

## Rebuild

```sh
.conda-envs/opencap/bin/python -m medpose.ui.build --output medpose/ui/demo.html
```

The builder uses existing locked NumPy/OpenCV and the project's existing ffmpeg
executable. Temporary rendered media stays under the repository's pytest cache
and is removed after embedding. The runtime UI has no third-party dependencies.

## Validation actually run

Unit tests verify the generated Trial, hand-computed schematic geometry, embedded
video decoding/dimensions, offline resource restrictions and provenance labels.
Headless Chrome tests at **1440 × 1100** and **390 × 844** exercise real DOM actions:
video initialization, playback controls, scrubbing/frame synchronization, joint
and camera selection, local File imports, comparison warnings, benchmark warnings,
report export/escaping, clearing stale video, empty Trial and malformed inputs.
Desktop and full-page mobile screenshots were inspected; a narrow-screen overflow
and a cropped video were fixed before final verification.

The test browser is controlled over OS pipes, not a network debugging port.
Tests trap page network APIs and inspect resource entries: zero network API calls
and zero HTTP(S) resources were observed in the tested flows. CSP independently
sets `connect-src 'none'` and only permits embedded/local blob media. This checks
the application's flows, not every possible browser-internal service.

Chrome and Node are test tools, not UI runtime dependencies. Initial browser-test
harness failures were fixed (escaping embedded script text and setting an actual
mobile viewport); assertions were retained. Automatic pytest plugin loading is
disabled because the shell exposes unrelated ROS plugins.

Final command:
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest -k 'not test_bundled_iphone_profile_still_loads' --basetemp=.pytest_cache/ui-integration
```
Actual final output:
```text
====================== 235 passed, 1 deselected in 24.54s ======================
```
The unchanged deselected test reads a non-synthetic bundled iPhone profile, which
is outside this task's synthetic-only scope. No patient recordings or forbidden
data directories were read. Work remains on `research-agent/build`; no push or
merge was performed.

## Single-picture review

Use **Open picture** to choose a local PNG, JPEG, WebP, AVIF or BMP. Decoding stays
in the browser. The image is fitted without cropping, its dimensions are shown,
and video playback/speed controls are disabled. SVG is deliberately unsupported.
Unsupported or malformed images leave the current review unchanged.

Opening a picture from the synthetic demo clears the unrelated invented Trial and
video. With only a picture, no joint angles, model segments, gait events, timing or
ranges are invented. You can load a separate matching Trial and select its frame
manually. The UI explicitly states that the Trial and picture were loaded
separately, and the image did not generate the Trial. Replacing a picture releases
its local object URL; resetting the demo restores video controls.

Picture-only export contains review notes, dimensions and limitations. It does
not embed the image. Export with a separate Trial retains the Trial curves and
explicitly states that they were not estimated from the picture. No pose model,
clinical threshold or inference adapter was added.

Additional browser checks use a generated 32 × 24 PNG, never a real recording.
They test import, clearing demo data, dimensions, disabled playback, no fabricated
range, report limits, invalid-format/decode rejection, manual Trial-frame matching,
reset behavior and no remote resource/network API use on desktop and mobile.
The first rerun exposed a test-fixture FileList reused after the input had cleared;
the test now performs a fresh file selection, retaining all feature assertions.

Command:
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_ui.py --basetemp=.pytest_cache/picture-ui
```
Actual output:
```text
============================== 4 passed in 7.63s ===============================
```
These are the four existing UI test cases with expanded browser workflows, not
four additional test cases. The full Python regression suite was not repeated for
this UI-only change.
