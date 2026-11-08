# Local picture and video inference

The earlier single-file Trial viewer only displayed pictures. Pose Studio adds
actual inference using MediaPipe Pose Landmarker, running in a browser worker on
CPU. It supports one picture, up to 20 separately reviewed pictures, and one
single-person video up to 120 seconds / 250 MB. Videos are sampled at 10 Hz.
Pictures show a static estimated pose; they never create measured movement.

## Start

```sh
.conda-envs/opencap/bin/python tools/setup_pose.py
.conda-envs/opencap/bin/python -m medpose.ui.serve
```

Setup downloads pinned public models and the runtime, checks their digests, and
creates the synthetic stationary-person video. It adds no Python packages and
leaves the OpenCap environment and upstream checkouts unchanged. Runtime version
is `@mediapipe/tasks-vision` 0.10.21. Weights are Google's `float16/1` releases.

Open http://127.0.0.1:8765/ and choose **Open picture(s)** or **Open video**, then
**Estimate pose**. The synthetic-picture and synthetic-video buttons run their
fixtures immediately. Drag the 3D skeleton to rotate; video playback and curves
follow the selected sample. **Clear** releases local object URLs. **Cancel** kills
the worker and discards partial output. Model errors are shown, never replaced by
synthetic results. The legacy demo is accessible under **Trial review & demo**.

## Models

| Option | Download | Use |
|---|---:|---|
| Lite | 5,777,746 bytes | Lower compute requirement |
| Full | 9,398,198 bytes | Balanced default |
| Heavy | 30,664,242 bytes | Highest-capacity MediaPipe option provided here |

All three were downloaded and verified locally. The Models panel reports
installation status and offers a download action for missing/corrupt weights.
Checksums and exact model URLs are in `medpose/ui/models.py`; the runtime archive
SHA-512 is in `tools/setup_pose.py`. Downloads are explicit setup operations.
The highest-capacity label is not a claim that Heavy is best for CP or every image.
No benchmark comparing these models on CP was performed.

Public documentation and sources:
- [Google Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- [Pinned runtime package](https://www.npmjs.com/package/@mediapipe/tasks-vision/v/0.10.21)
- [MediaPipe source](https://github.com/google-ai-edge/mediapipe)

The runtime's Apache 2.0 notice is retained under `medpose/ui/licenses/`.
Downloaded runtime and weights stay under ignored `.conda-envs/pose-web/`, not git.

## Output and boundaries

The left overlay shows image landmarks. The rotatable model uses the returned
33 hip-centred world landmarks, not generic limb lengths invented from the demo.
Depth and scale are model estimates, not a calibrated ground coordinate frame or
an image of the person's bones. No anatomical mesh is generated.

Landmark knee and elbow bends use `180° - acos(dot(a,b)/(|a||b|))`, with vectors
from the middle landmark to the other two points. A straight limb is 0°, an
orthogonal pair is 90°. These are geometric bends, not OpenSim anatomical joint
coordinates. Exported Trial names deliberately start with `landmark_` so clinical
rules for `knee_angle_*` or `hip_flexion_*` cannot silently consume them.

Visibility/presence gating is 0.5, explicitly **provisional display gating**, not a
clinical threshold. The pinned runtime sometimes omits presence; visibility alone
is used then, and missing/nonfinite visibility means no confidence. Low-score
segments are dashed; their bends are missing. No detection or multiple detected
people yields an empty frame, never a made-up pose. The detector is configured to
look for up to two people to detect ambiguity; it is not reliable identity
tracking, so input must contain one person. It may still miss another person.

**Export results** stores raw landmarks, source model, model revision, sample
rate, supports/context and limitations in JSON. **Export Trial** stores the four
geometric bends and their model confidence in the shared schema. Choose the
correct phone/webcam source and enter supports before exporting. Neither export
contains media or filenames. Error remains **not measured**. A confidence score
is not an estimate of angle accuracy. Clinical patterns, gait events, ground
contact, forces, diagnosis and exercise prescriptions are not inferred here.

## Local transport and privacy

The browser fetches the runtime and weights from a server bound only to
127.0.0.1. This is an intentional change from the original zero-request standalone
viewer: the new studio uses loopback asset requests, while the original viewer
retains its no-network behavior. Neither makes external inference calls. Selected
media is decoded in the browser and transferred only to its local worker.

The server uses an explicit route allowlist, never serves the repository or
patient directories, accepts no media-upload endpoint, and suppresses request
logs. Model installation accepts only a fixed model ID with matching Origin/Host,
a custom action header and an empty request body; it downloads only pinned public
weights. A CSP limits application and worker fetches to the same origin. Tests
block external DNS, capture browser/worker requests and assert local GET-only
traffic during inference. No account, cloud endpoint or telemetry is configured.

## Advanced body-mesh / future cloud route

The existing OpenCap/WHAM wrappers and approximately 6.1 GB of public WHAM/HMR2/
ViTPose/YOLO weights remain in place. They are separate from this landmark studio.
The UI exposes their setup checklist but does not pretend this route is runnable.
No cloud compute was provisioned, no spending started and no media uploaded.

The user supplied `SMPL_python_v.1.1.0.zip` after accepting its licence. The
male, female and neutral models are now installed under `.conda-envs/smpl/`, with
SHA-256 provenance and the supplied licence. Model-file links connect this store
to `opencap-monocular/WHAM/dataset/body_models/smpl/`, using the filenames that
WHAM, SLAMHR and OpenCap's optimisation code expect. No licensed assets enter git.
The archive includes neutral, so no separate SMPLify download is needed for it.

To reproduce the import from a user-licensed archive:

```sh
.conda-envs/opencap/bin/python -m medpose.engine.smpl_assets install /path/to/SMPL_python_v.1.1.0.zip
.conda-envs/opencap/bin/python -m medpose.engine.smpl_assets status
```

The installer selects only the three known model members and the licence; it
never runs archive scripts or unpickles during extraction, and refuses to replace
unrelated installed models. All models loaded through the existing `smplx`
dependency and generated two synthetic poses on CPU: 6,890 vertices and 13,776
faces each. The 300-component archive works with the 10 betas used by the existing
pipeline. A separate CUDA forward pass with the neutral model also passed on the
NVIDIA GeForce 940MX (4,096 MiB, compute capability 5.0), using PyTorch 2.1.2+cu118.
Peak tensor allocation for that small SMPL-only check was 27.8 MiB; this is **not**
a memory estimate for full WHAM inference.

Full inference remains unverified: `J_regressor_wham.npy`,
`J_regressor_h36m.npy`, `J_regressor_feet.npy` and the `ultralytics` dependency
are missing, and other full-pipeline requirements have not been validated. The
local UI now reads `/api/advanced` to show actual SMPL installation status while
keeping the full body-mesh pipeline marked unavailable. MediaPipe picture/video
inference remains separate from SMPL; installing geometry does not fit it to a
photograph. No patient photograph or recording was processed.

- [SMPL registration](https://smpl.is.tue.mpg.de/register.php)
- [SMPL model licence](https://smpl.is.tue.mpg.de/modellicense.html)
- [WHAM registration requirements](https://github.com/yohanshin/WHAM#registration)
- See `OPENCAP_START.md` for existing calibration and pipeline requirements.

## Verification

Actual browser inference ran on the AI-generated adult fixture, and a stationary
video made from it. No supplied photograph or real recording was processed.
Tests cover each model, blank input, invalid images, picture switching, video
sampling and seeking, cancellation, exports, boundary geometry, checksum failure,
server path/installation controls and desktop/mobile layout.

The first checks exposed a runtime MIME type error and a mobile paused-video
frame decode issue. Both were fixed; assertions were retained. These checks show
software function on synthetic inputs, not clinical or population accuracy.

Final verification on 2026-11-08:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest -k 'not test_bundled_iphone_profile_still_loads' --basetemp=.pytest_cache/pose-complete
====================== 241 passed, 1 deselected in 55.02s ======================
```

The deselected calibration fixture accesses a real recording; it was excluded
under the synthetic-only project rule. No test assertion was weakened. The
standalone viewer's setup button shows launch instructions and performs no network
request, retaining its strict offline tests. Actual inference and all three model
variants passed desktop/mobile browser checks with external DNS blocked.

The earlier browser-access blocker was superseded by sign-in, followed by a
temporary 504 error from the official download host. Both blockers were resolved
for model installation when the user supplied the downloaded archive. No
authentication credentials were extracted.

Picture-flow clarification: **Open picture(s)** only loads media. **Estimate pose**
is now a prominent green button, with explicit next-step instructions in the
status banner and empty model panel. The old standalone viewer directs users to
Pose Studio instead of implying it performs inference. The local Studio was
reachable and reported all three models installed during this check.

After this clarification, the targeted UI and actual-model checks completed with:
`10 passed in 23.44s` using `tests/test_ui.py tests/test_pose_studio.py` and
`.pytest_cache/pose-next-step`. Synthetic inputs only; no supplied picture processed.

SMPL integration verification (2026-11-08):

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_smpl_assets.py tests/test_pose_studio.py tests/test_engine_wrappers.py --basetemp=.pytest_cache/smpl-integration
======================= 28 passed, 19 warnings in 41.00s =======================
```

Warnings come from the upstream Python-2-era Chumpy/SciPy/NumPy model loader.
The separate synthetic GPU calculation completed successfully. The full
repository suite was not repeated for this asset-integration change.
