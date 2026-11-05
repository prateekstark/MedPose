# MedPose

Offline 3D movement analysis from ordinary cameras, for cerebral palsy research. MedPose is being rebuilt around a shared results format, with OpenCap Core and Monocular as its current pipelines and as comparison baselines.

> **Research use only.** MedPose isn't clinically validated for CP assessment. Don't use its output for diagnosis or treatment decisions. Validate each measure against an accepted reference method on representative CP recordings.

Read [docs/DATA_POLICY.md](docs/DATA_POLICY.md) before recording anyone. Scientific limits and the Android setup are in [OPENCAP_START.md](OPENCAP_START.md).

## Status
| Mode | State |
|---|---|
| 2+ cameras (OpenCap Core) | Runs on upstream's healthy iPhone test clip. **Not yet run on real Android footage** |
| 1 Android phone or webcam (OpenCap Monocular) | Wrapper exists, but **has never completed a run here**: needs the SMPL licence, missing dependencies and a better GPU |
| Trial format, gait/exercise summaries, change checks, prompts, reports and benchmarks | Research implementations with synthetic tests; thresholds provisional, error table empty, no validation on people |

## Layout
| Path | Purpose |
|---|---|
| `medpose/capture/video.py` | Makes phone and webcam video upright and constant-rate |
| `medpose/capture/intrinsics.py` | Checkerboard calibration, with a quality gate |
| `medpose/subject.py` | Requires measured height and mass; no adult defaults |
| `medpose/engine/` | `core_stage`, `core_run` and `mono_run` wrappers around OpenCap |
| `medpose/viewer/viewer.html` | Offline Trial viewer with schematic stick figures and curves; uploads nothing |
| `tests/` | Test suite, using synthetic footage only |
| `tools/download_public_weights.py` | Fetches public checkpoints, excluding SMPL |
| `upstream.lock.json`, `patches/` | Pinned upstream commits and our local edits to them |
| `env/opencap-py39.lock.txt` | Package versions of the working environment |
| `docs/research_agent_brief.md` | Fact-check and novelty-check brief for an external research agent |
| `archive/` | The earlier MediaPipe prototype and original scripts. Don't use them for CP measurement |

`opencap-core/` and `opencap-monocular/` are upstream checkouts, ignored by git. They stay in place because the environment has editable installs pointing into them.

## Running
Use the project environment for everything, and run from this folder:

```bash
.conda-envs/opencap/bin/python -m pytest
```

Calibrate each camera for its exact lens and video mode:

```bash
.conda-envs/opencap/bin/python -m medpose.capture.intrinsics checkerboard.mp4 --corners 9 6 --square-mm 25 --output cam.pickle
```

Calibration is refused if it fails the quality gate, and nothing is saved. The default limits (1 px reprojection error, 50% image coverage) are rules of thumb that can be changed with `--max-rms` and `--min-coverage`.

The full multi-camera and single-camera steps are in [OPENCAP_START.md](OPENCAP_START.md).

## Video handling
OpenCap was built around iPhone recordings, so every other video is normalised before OpenCap sees it. Orientation is baked into the pixels, variable frame rates become constant, and the result is an upright `.mov` with frequent keyframes. Any format ffmpeg reads works, including webcam `.mkv`, `.avi`, `.webm` and MJPEG.

Record the checkerboard and movement videos with the same camera, lens, resolution and orientation. Turn off a webcam's autofocus before calibrating, because refocusing changes the intrinsics.

## Licences
OpenCap Core is Apache-2.0. OpenCap Monocular is PolyForm Noncommercial, and SMPL has its own non-commercial licence that you must accept yourself. MedPose's own code has no licence yet; choose one before any public release.


## Open the review UI

Open [the offline demo](medpose/ui/demo.html) directly in Chrome or another modern
browser. It is a single HTML file with its video and Trial data embedded; no server
or internet connection is needed. Use Play, scrub the timeline, rotate the model,
select a joint, compare local Trial files or export an offline report.

The sample is **synthetic**: its rendered walker video and motion curves come from
the same invented data. No pose was recovered from video and no clinical accuracy
is demonstrated. Attaching another local video does not run inference. Gait,
exercise, MDC and pattern analysis remain in the Python modules; the browser does
not fabricate clinical analysis. See [UI notes](docs/research/UI_DEMO.md).

To rebuild the embedded demo from source:

```bash
.conda-envs/opencap/bin/python -m medpose.ui.build --output medpose/ui/demo.html
```

## Research analysis layer

Read [the build report](docs/research/BUILD_REPORT.md) for package commits, actual
Python 3.9 test output, formulas, threshold status and remaining validation gaps.
`medpose.trial.Trial` is the engine-independent input; schema 2 adds optional
world-frame heel/toe sole positions and a ground plane. Schema 1 remains readable.
`load_opensim_mot` requires explicit acquisition metadata and unit declarations;
it does not infer contacts or a participant's body geometry.

The error table ships empty. Pattern thresholds ship disabled; gait and exercise
segmentation settings must be supplied explicitly and remain provisional.
Reports are static standalone HTML/SVG. Open `medpose/viewer/viewer.html` directly
and select local Trial files; the viewer no longer accepts legacy OpenCap visualizer
JSON or `?src=` URLs. Its simplified stick figure is not a full-body reconstruction.
No model adapter, dataset loader, kinetics or original GDI was added.
