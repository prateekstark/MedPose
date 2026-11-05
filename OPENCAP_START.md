# OpenCap Workflows

Both official code paths are checked out in this project:

- [`opencap-monocular/`](opencap-monocular/) — one-video biomechanics pipeline.
- [`opencap-core/`](opencap-core/) — calibrated multi-camera biomechanics pipeline.

The earlier prototype scripts are in `archive/`; OpenCap is the intended processing base.

## How well does Monocular work?

The 2026 OpenCap Monocular paper reports mean errors of **4.8° for rotational kinematics** and **3.4 cm for pelvis translation**, averaged across 10 healthy adults doing walking, squats, and sit-to-stand, against marker-based motion capture. Its pose optimization reduced the monocular model's drift, but pelvis translation still drifted more than the two-camera OpenCap result in the paper. These results are promising for those tasks in that study; they do not establish accuracy for CP gait or for an individual patient. [Paper](https://arxiv.org/abs/2603.24733)

The official Monocular guidance lists walking, sit-to-stand, and squats as validated activities. It recommends a static camera around 30–45° from the participant, full body visible, and avoiding long occlusions. The public recording app is iOS-only; the standalone local code takes a video path and can be adapted for Android footage if the correct camera intrinsics and calibration inputs are supplied. [Best practices](https://www.opencap.ai/best-practices?variant=monocular) · [Standalone code](https://github.com/utahmobl/opencap-monocular)

## Experimental extension: dense monocular depth

We can test a metric-depth model as an additional soft constraint on OpenCap Monocular's pose optimization. OpenCap already initializes 3D body pose with WHAM, then optimizes pose/camera and applies biomechanical constraints; dense depth would add another cue rather than replace that pipeline. [OpenCap Monocular method](https://arxiv.org/abs/2603.24733)

Candidate models such as [UniDepth](https://arxiv.org/abs/2403.18913) and [Metric3D v2](https://arxiv.org/abs/2404.15506) estimate metric depth from one image, but their reported generalization is not validation on CP patients or their gait. Relative-depth models do not give absolute scale without an added scale reference. Dense depth is also a visible-surface map: a predicted depth at a knee or ankle pixel is not automatically the anatomical joint center, and depth can flicker between frames.

The experiment should add depth as a confidence-weighted, temporally smoothed loss against a rendered body surface, then compare baseline and augmented results with synchronized Android views or marker-based reference data. Keep calibrated multi-camera reconstruction as the geometric reference; don't treat monocular depth predictions as ground truth.

For CP, neither workflow should be assumed clinically accurate without task- and measure-specific validation. In a study of children with CP, a different markerless system captured some gait measures well but missed pelvic-tilt and transverse-plane hip deviations. That is a warning about the clinical problem, not a direct accuracy result for OpenCap. [CP gait study](https://pubmed.ncbi.nlm.nih.gov/38789587/)

## Workflow A: one Android video with OpenCap Monocular

Use the local Monocular pipeline; the hosted iOS recording app is not the Android route. The local wrapper accepts one video plus subject metadata, a camera-calibration file, and camera intrinsics. Android footage needs intrinsics for the exact phone camera, resolution, and lens used; do not rely on the bundled iPhone defaults. The wrapper normalizes the video (upright, constant frame rate, `.mov`; see [`video.py`](medpose/capture/video.py)) into `patient_data/normalized/` and refuses to run if its upright size differs from the calibrated size.

Once model files and dependencies are in place, run it from this project with:

```bash
(.conda-envs/opencap/bin/python -m medpose.capture.intrinsics checkerboard.mp4 \
  --corners 9 6 --square-mm 25 --output android_cameraIntrinsics.pickle)

.conda-envs/opencap/bin/python -m medpose.engine.mono_run walking.mp4 \
  --metadata subject.yaml \
  --intrinsics android_cameraIntrinsics.pickle --activity walking
```

The checkerboard dimensions mean inner corners, and the square edge must be measured. Record the calibration video with the same rear-camera lens, resolution, frame rate, and orientation as the movement video. Holding the phone sideways for one and upright for the other is caught as a size mismatch. For a moving camera, also provide the WHAM camera-motion calibration with `--calibration` and add `--estimate-world-camera`.

The subject YAML needs the measured values for the person being analyzed:

```yaml
height_m: 1.70  # replace with measured height in meters
mass_kg: 70.0   # replace with measured mass in kilograms
sex: female     # male or female, as expected by the selected model
```

The outputs, including `mono.json`, are written under the Monocular checkout's `results/` folder. Open that JSON in [`viewer.html`](medpose/viewer/viewer.html) for playback and different camera angles.

The source repository is licensed under **PolyForm Noncommercial 1.0**. Public WHAM, ViTPose, and Core weights are available locally. Required SMPL body-model files have a separate license that must be reviewed and accepted by the user before downloading; those files are excluded from the downloader.

Official local setup is tested on Ubuntu 20.04/22.04, Python 3.9, CUDA 11.8 or 12.x, and NVIDIA driver 520 or newer. It also needs OpenSim, PyTorch, WHAM/ViTPose dependencies, and camera calibration inputs. [Install guide](https://github.com/utahmobl/opencap-monocular/blob/main/installation/INSTALL_SLIM.md)

## Workflow B: multiple Android videos with OpenCap Core

Use at least two views; three or four can reduce occlusion. Record fixed cameras at matching resolution and frame rate, and start them within a few seconds of each other; Core aligns the videos from the movement and a hand punch. The staging helper normalizes every camera to one constant frame rate, the slowest camera's when they differ (`--fps` overrides it), and writes `<trial_id>.mov`, which Core's file-name logic expects. Calibrate camera intrinsics and the shared rig with a checkerboard. OpenCap Core triangulates 3D markers and runs the OpenSim kinematics pipeline.

The local Core code can process near-synchronous video recorded with external devices, but its README says the easy external-video utilities are still under development. The current `main.py` expects an OpenCap session layout: `Data/<session>/Videos/Cam*/InputMedia/<trial>/`, `sessionMetadata.yaml`, and camera intrinsics. A local staging adapter is included. After preparing OpenCap-format metadata and calibrating the rig, stage the videos like this:

```bash
.conda-envs/opencap/bin/python -m medpose.engine.core_stage gait01 walking trial01 \
  --metadata sessionMetadata.yaml \
  --video Cam0=/path/view0.mp4 --video Cam1=/path/view1.mp4
```

For each Android phone model, create an intrinsics profile matching the exact lens and video mode. Set `iphoneModel` in `sessionMetadata.yaml` to those profile labels. Stage a synchronized checkerboard recording from every camera, then run Core's extrinsics-only calibration pass:

```bash
.conda-envs/opencap/bin/python -m medpose.capture.intrinsics checkerboard_cam0.mp4 \
  --corners 11 8 --square-mm 60 --core-camera-model android_cam0

.conda-envs/opencap/bin/python -m medpose.engine.core_stage gait01 checkerboard calibration01 \
  --metadata sessionMetadata.yaml \
  --video Cam0=/path/checkerboard_cam0.mp4 --video Cam1=/path/checkerboard_cam1.mp4

.conda-envs/opencap/bin/python -m medpose.engine.core_run gait01 checkerboard calibration01 \
  --cameras Cam0 Cam1 --extrinsics-trial
```

After rig calibration, stage a neutral standing trial (about 5 s, person still and upright) and run it with `--neutral`; this scales the OpenSim model to the person and is needed once per session before any movement trial:

```bash
.conda-envs/opencap/bin/python -m medpose.engine.core_stage gait01 neutral neutral \
  --metadata sessionMetadata.yaml \
  --video Cam0=/path/neutral_cam0.mp4 --video Cam1=/path/neutral_cam1.mp4

.conda-envs/opencap/bin/python -m medpose.engine.core_run gait01 neutral neutral \
  --cameras Cam0 Cam1 --neutral
```

Then stage the synchronized movement videos as another trial in the same session and run:

```bash
.conda-envs/opencap/bin/python -m medpose.engine.core_run gait01 walking trial01 \
  --cameras Cam0 Cam1
```

Outputs use OpenCap's generic folder names: joint angles in `Data/<session>/OpenSimData/Kinematics/<trial_id>.mot`, markers in `Data/<session>/MarkerData/PostAugmentation/`, and 3D viewer data in `Data/<session>/VisualizerJsons/<trial>/` (open it in `medpose/viewer/viewer.html`, or with `viewer.html?src=/path` when served by the CP Motion app).

A flat checkerboard has two mirror-image pose solutions per camera, and Core saves both during calibration. As OpenCap's server does, the wrapper passes both to the neutral trial (`--calibration-trial` names the checkerboard trial, default `calibration`), where Core keeps the pair that reconstructs the standing person best; the choice is printed and saved in `Videos/calibOptionSelections.json`.

**Keep every camera in the same slot for every trial.** If one phone's checkerboard video is staged as `Cam0` and its movement video as `Cam1`, nothing fails: Core produces a shrunken, tilted body and joint angles that are simply wrong. (OpenCap's own 2-camera test data has its calibration files named the other way round from its trial folders; paired as named, the joint angles were 30–60° off, and paired correctly they matched OpenCap's reference within 0.9° RMS.) The CP Motion app warns after the neutral trial when the reconstructed neck-to-ankle length is outside 65–90% of the entered height, which catches this.

The wrapper also handles three things the local environment differs in from OpenCap's Docker image: it puts MMPose's `configs/` (from the ViTPose checkout) on the import path for HRNet, it adds the environment's `bin/` to `PATH` so Core's `opensim-cmd` scaling call works without activating the environment, and the staging normalization writes a keyframe every 0.1 s, because Core grabs calibration and neutral stills by seeking to keyframes.

The wrapper selects OpenCap Core's HRNet WholeBody pose detector through MMPose. Android phone intrinsics and shared-rig calibration must be supplied; staging videos alone does not calibrate the rig. [Core source and install guide](https://github.com/opencap-org/opencap-core)

The Core repository is Apache-2.0 licensed. Its documented local inference setup uses Python 3.9 and CUDA/GPU dependencies. OpenCap Processing utilities are a separate environment, currently documented with Python 3.11 and OpenSim 4.5. [Processing utilities](https://github.com/opencap-org/opencap-processing)

## Live camera processing

Neither configured workflow is real-time. Both analyze recorded video offline, and the 940MX's 4 GB GPU is an older, low-memory device; processing is expected to be slower than the video's playback rate. `viewer.html` plays the completed motion output interactively, but it does not track a live camera feed.

## Setup status on this machine

- Both official source repositories and their Git submodules are present locally.
- The existing `starkml` environment contains PyTorch `2.14.1a0+git5c48869`, built with CUDA 12.9, and supports the GPU's `sm_50` architecture. A GPU-enabled check reported CUDA available and one device; a later restricted shell could not access `/dev/nvidia0`. The NVIDIA GeForce 940MX has 4 GB VRAM and driver 580.178.04.
- `starkml` uses Python 3.14.7 and NumPy 2.5.2, and does not have OpenSim. OpenCap Monocular's documented setup requires Python 3.9, NumPy 1.21, and OpenSim 4.4; OpenCap Core also uses older pinned dependencies. So the installed PyTorch itself is usable, but the environment as a whole cannot run either official workflow as-is. The user asked to use `starkml`, so its packages have been left untouched.
- With the user's approval, a project-local `.conda-envs/opencap` environment has been created with Python 3.9.25, NumPy 1.21.6, and OpenSim 4.4. It has PyTorch 2.1.2+cu118, MMCV 1.7.2, ViTPose/MMPose 0.24.0, MMDetection 2.28.2, and TensorFlow 2.9.3. PyTorch sees the NVIDIA GeForce 940MX. TensorFlow currently runs on CPU because the separate CUDA 11.2/cuDNN 8.1 runtime requires NVIDIA EULA review and acceptance; those runtime packages were removed after the installer displayed that notice.
- Public WHAM, ViTPose, Core Faster R-CNN, and HRNet checkpoints have downloaded and passed size checks. SMPL body-model files remain excluded pending the user's separate license acceptance.
- A first separate Python 3.9/OpenSim install was attempted, but the package-server connection reset during OpenSim download. That partial environment was removed; the later project-local environment above was installed successfully.
- The camera-intrinsics catalog from OpenCap Core is copied into the Monocular checkout. It contains iPhone/iPad entries, not Android calibration profiles. Android input therefore still needs camera/lens-specific calibration at the chosen resolution and frame rate.
- OpenCap Monocular's environment is missing `ultralytics`, which its full `installation/requirements.txt` lists; Monocular stops at import until it is installed.
- [`viewer.html`](medpose/viewer/viewer.html) is a standalone, local interactive viewer for OpenCap visualizer JSON outputs. It has been opened in Codex, but not exercised with generated motion data.
- No patient videos have been used or uploaded. SMPL body-model files and the official WHAM fetch script require account credentials and separate license acceptance; those files have not been downloaded.

The compatible inference dependencies and public checkpoints are in place, but full Monocular inference is not yet ready: the body-model step needs your separate license acceptance. The multi-camera route still needs Android camera calibration and synchronization. The current tools process recorded videos offline, not live camera streams. Before using either output for CP assessment, compare the exact measures against a trusted reference method on representative CP recordings.

## Privacy note

The hosted OpenCap service uploads videos for cloud processing. Its current terms say submitted video is visible to the development team and may be used for algorithm development; the investigator must obtain participant consent, and the hosted service is for noncommercial academic/nonprofit research and education. Use local processing for patient footage unless the applicable consent and institutional approvals explicitly allow the hosted route. [Terms](https://www.opencap.ai/terms-conditions)
