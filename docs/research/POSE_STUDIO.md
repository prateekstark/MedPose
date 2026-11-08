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
an image of the person's bones. This landmark workspace generates no body surface; use `/mesh` for that output.

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
patient directories, and suppresses request logs. The landmark workspace has no
media upload. The separate mesh workspace uses the protected loopback-only
`/api/mesh` endpoint described below. Model installation accepts only a fixed model ID with matching Origin/Host,
a custom action header and an empty request body; it downloads only pinned public
weights. A CSP limits application and worker fetches to the same origin. Tests
block external DNS, capture browser/worker requests and assert local GET-only
traffic during inference. No account, cloud endpoint or telemetry is configured.

## Advanced body-mesh / future cloud route

The existing OpenCap/WHAM wrappers and approximately 6.1 GB of public WHAM/HMR2/
ViTPose/YOLO weights remain in place. They are separate from this landmark studio.
The `/mesh` workspace connects local HMR2 and WHAM adapters to the browser.
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

### Body mesh studio

Open `/mesh`, select a picture or video, choose the model, and click **Generate
mesh**. HMR2 generates 6,890-vertex SMPL surfaces from detected person crops.
For videos it processes frames independently. WHAM estimates a sequence using
ViTPose, HMR2 features and its temporal network. Its adapter requires one visible
person, at least 30 frames, and a stationary camera. It does not run SLAM or claim
calibrated global motion. Both routes limit video to 120 frames and uploads to
64 MB. Partial/multiple detections remain missing for HMR2; WHAM rejects a clip
without a continuous single-person detection. This is not reliable identity
tracking. No clinical joint-angle Trial is exported from mesh predictions.

The browser passes only selected media to local Python. Linux anonymous memory
files transfer media to a child process; no source media is written to disk.
Results are kept in RAM, limited to two jobs, and can be exported explicitly.
Cancel terminates the worker; a thirty-minute timeout bounds a job. Exact Origin,
Host and action headers are required. Inference disables outbound socket
connections and automatic dependency downloads. Model setup is separate.
Generic body-model estimates are not anatomical scans; pediatric proportions,
clothing, occlusion, depth and scale remain limitations, not measured accuracy.

Install missing auxiliary assets and adapter packages in the existing environment:

```sh
.conda-envs/opencap/bin/python tools/setup_advanced.py
```

The script pins Ultralytics 8.0.100, seaborn 0.12.2, py-cpuinfo 9.0.0,
thop 0.1.1.post2209072238 and sentry-sdk 1.45.0 without upgrading the OpenCap
scientific stack. Telemetry is disabled and network calls blocked at inference.
The official WHAM auxiliary archive has SHA-256
`533a8b6b05ba3bc3d35c1dcaec757a4164a669625116a16d30d7642f3b200c8c`.
Only seven known regular-file members are extracted; no archive code executes.
Public large checkpoints remain reproducible with `tools/download_public_weights.py`.
See [WHAM](https://github.com/yohanshin/WHAM) and
[4D-Humans](https://github.com/shubham-goel/4D-Humans) for upstream methods.

A real HMR2 run on the synthetic adult produced finite geometry on the 940MX:
24.04 seconds including detection/loading, 2,688.9 MiB peak tensor allocation.
WHAM also produced 30 finite 6,890-vertex meshes from a synthetic 3-second
stationary-person video: 246.68 seconds, 2,670.1 MiB peak GPU allocation.
The **Try synthetic video** button loads this generated clip. It is not a walking
recording or an accuracy benchmark. These are functionality checks only.
Full calibrated OpenCap fitting remains separate and unverified; it requires
matching camera calibration and subject metadata. No cloud account, spending,
or upload has been configured.

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

Body-mesh checks added on 2026-11-08 cover finite geometry, invalid media,
request-origin guards, upload limits, concurrent-job rejection, cancellation,
anonymous-memory file transfer, archive checksums and real browser-to-GPU
inference. Run the optional heavier check with:

```sh
MEDPOSE_TEST_MESH=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda-envs/opencap/bin/python -m pytest tests/test_mesh_studio.py::test_mesh_browser_with_installed_models
```

It uses only the committed AI-generated adult, checks a visible mesh and export,
and verifies that browser traffic stays on the loopback server. No personal
pictures or patient recordings are used. Model files and test output stay ignored.
