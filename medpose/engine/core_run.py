#!/usr/bin/env python3
"""Process a calibrated multi-camera OpenCap Core session stored locally."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path

from medpose.paths import CORE_ROOT
from medpose.subject import SubjectError, load_core_session


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenCap Core on an already staged and calibrated session."
    )
    parser.add_argument("session", help="Session folder name under opencap-core/Data")
    parser.add_argument("trial", help="Trial folder name under each camera's InputMedia")
    parser.add_argument("trial_id", help="Video basename shared by all selected cameras")
    parser.add_argument("--cameras", nargs="+", required=True,
                        help="At least two camera names, e.g. Cam0 Cam1 Cam2")
    parser.add_argument("--resolution", default="default",
                        help="OpenCap Core pose-detection resolution setting")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extrinsics-trial", action="store_true",
                      help="Run only the rig-calibration pass for a checkerboard trial")
    mode.add_argument("--neutral", action="store_true",
                      help="Neutral standing trial: scale the OpenSim model to the person; run "
                           "once per session after calibration and before movement trials")
    parser.add_argument("--calibration-trial", default="calibration", metavar="TRIAL",
                        help="With --neutral: the checkerboard trial whose two calibration "
                             "solutions per camera Core chooses between (default: calibration)")
    args = parser.parse_args()

    if len(args.cameras) < 2:
        parser.error("multi-camera 3D reconstruction needs at least two calibrated views")
    session_dir = CORE_ROOT / "Data" / args.session
    metadata = session_dir / "sessionMetadata.yaml"
    if not session_dir.is_dir() or not metadata.is_file():
        parser.error(f"Expected a prepared session with metadata at {metadata}")
    try:
        load_core_session(metadata)
    except SubjectError as e:
        parser.error(str(e))
    for cam in args.cameras:
        trial_dir = session_dir / "Videos" / cam / "InputMedia" / args.trial
        matches = list(trial_dir.glob(args.trial_id + ".*"))
        if not matches:
            parser.error(f"No video for {cam} in {trial_dir}; expected basename {args.trial_id}")

    # Core's HRNet code imports MMPose's dataset definitions as `configs._base_...`; its
    # Docker image runs from an MMPose source checkout that has them at the top level.
    # Here MMPose is the ViTPose checkout installed in editable mode: put its root on the path.
    try:
        import mmpose
    except ImportError as e:
        parser.error(f"MMPose is not installed in this environment ({e}); run this with "
                     ".conda-envs/opencap/bin/python")
    mmpose_root = Path(mmpose.__file__).resolve().parent.parent
    if not (mmpose_root / "configs" / "_base_" / "datasets" / "coco_wholebody.py").is_file():
        parser.error(f"MMPose's configs folder was not found next to {mmpose_root / 'mmpose'}; "
                     "Core's pose detection needs the MMPose (ViTPose) source checkout")
    sys.path.append(str(mmpose_root))

    os.chdir(CORE_ROOT)
    sys.path.insert(0, str(CORE_ROOT))
    # Core's utils.py asks for an app.opencap.ai login at import time, although local
    # processing never calls the web API. A placeholder skips that prompt; it is not a
    # credential and is never sent anywhere by main().
    os.environ.setdefault("API_TOKEN", "local-processing-only")
    # Core runs OpenSim's scaling as the shell command `opensim-cmd`, which lives next to
    # this environment's python and isn't on PATH unless the environment is activated.
    # Appended, so system tools Core also calls (ffmpeg) keep their usual versions.
    os.environ["PATH"] = os.pathsep.join(
        [os.environ.get("PATH", ""), str(Path(sys.executable).parent)])
    import main as opencap_core
    import utils as opencap_utils

    # A flat checkerboard gives two mirror-image pose solutions per camera, and Core saves
    # both. OpenCap's server passes them to the neutral trial, where Core picks the pair that
    # reconstructs the standing person best and keeps it for later trials; without them
    # Core silently uses the first solution, which can put the scene upside down or skewed.
    calibration_options = None
    if args.neutral:
        calibration_options = {}
        for cam in args.cameras:
            calib_dir = session_dir / "Videos" / cam / "InputMedia" / args.calibration_trial
            solutions = [calib_dir / f"cameraIntrinsicsExtrinsics_soln{k}.pickle" for k in (0, 1)]
            missing = [str(f) for f in solutions if not f.is_file()]
            if missing:
                parser.error(f"Calibration solutions not found: {', '.join(missing)}; run the "
                             f"'{args.calibration_trial}' trial with --extrinsics-trial first")
            try:
                calibration_options[cam] = [opencap_utils.loadCameraParameters(str(f))
                                            for f in solutions]
            except (OSError, EOFError, pickle.UnpicklingError) as e:
                parser.error(f"Could not read the calibration of {cam}: {e}")

    # Upstream Core hard-codes the MMPose model directory for one developer machine.
    # Point it at this project's local model folder instead.
    model_dir = CORE_ROOT / "models"
    opencap_core.getMMposeDirectory = lambda isDocker=False: str(model_dir)
    # Core raises many exception types, including bare Exception for bad data, so anything it
    # raises counts as a failed trial. The traceback stays, because it's the only diagnostic.
    try:
        opencap_core.main(
            sessionName=args.session,
            trialName=args.trial,
            trial_id=args.trial_id,
            cameras_to_use=args.cameras,
            poseDetector="mmpose",
            resolutionPoseDetection=args.resolution,
            extrinsicsTrial=args.extrinsics_trial,
            scaleModel=args.neutral,
            calibrationOptions=calibration_options,
            # outputs go to MarkerData/ and OpenSimData/ rather than detector-named
            # subfolders, so they're found at the same paths whatever the settings
            genericFolderNames=True,
        )
    except Exception as e:
        traceback.print_exc()
        parser.exit(1, f"OpenCap Core failed on {args.session}/{args.trial} ({type(e).__name__}: {e}). "
                       f"Don't use any output from this trial.\n")
    if args.extrinsics_trial:
        print(f"Calibration finished: cameraIntrinsicsExtrinsics.pickle in each "
              f"{session_dir}/Videos/Cam*/ folder.")
    elif args.neutral:
        try:
            choice = (session_dir / "Videos" / "calibOptionSelections.json").read_text().strip()
        except OSError:
            choice = "(not recorded)"
        print(f"Scaling finished: model in {session_dir}/OpenSimData/Model/. "
              f"Calibration solution chosen per camera: {choice}")
    else:
        print(f"OpenCap Core finished: joint angles in {session_dir}/OpenSimData/Kinematics/"
              f"{args.trial_id}.mot, markers in {session_dir}/MarkerData/PostAugmentation/, "
              f"3D viewer data in {session_dir}/VisualizerJsons/{args.trial}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
