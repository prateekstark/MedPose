"""Build a self-contained offline UI with synthetic video and shared Trial data.

No video-to-pose inference is performed. The video and Trial are rendered from the
same invented motion curves. No normative or human-accuracy claim is intended.
Run: .conda-envs/opencap/bin/python -m medpose.ui.build --output medpose/ui/demo.html
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from medpose.trial import Trial
from medpose.report import render_report

ROOT = Path(__file__).resolve().parents[2]


def synthetic_trial():
    fps = 30
    time = np.arange(240) / fps
    phase = 2 * np.pi * time / 1.6
    values = np.column_stack([
        12 + 24 * np.sin(phase), 12 + 24 * np.sin(phase + np.pi),
        12 + 46 * np.maximum(0, np.sin(phase - .5)) ** 2,
        12 + 40 * np.maximum(0, np.sin(phase + np.pi - .5)) ** 2,
        4 + 10 * np.sin(phase + .7), 4 + 10 * np.sin(phase + np.pi + .7),
    ])
    names = ("hip_flexion_l", "hip_flexion_r", "knee_angle_l", "knee_angle_r", "ankle_angle_l", "ankle_angle_r")
    return Trial(time, names, values, "1_phone", "synthetic-demo", "1", 1, fps,
                 supports=("Walker drawn in demo; not a participant recording",))


def joints(values):
    """Schematic segment geometry in arbitrary display units, not body dimensions."""
    result = {"pelvis": np.array([0., 0., 0.]), "chest": np.array([0., .58, 0.]),
              "head": np.array([0., .87, 0.])}
    for side, sign, hi, ki in (("l", -1, 0, 2), ("r", 1, 1, 3)):
        hip = np.array([sign * .14, 0., 0.])
        thigh = np.deg2rad(values[hi]); shank = thigh - np.deg2rad(values[ki])
        knee = hip + [0, -.49 * np.cos(thigh), .49 * np.sin(thigh)]
        ankle = knee + [0, -.47 * np.cos(shank), .47 * np.sin(shank)]
        result.update({"hip_" + side: hip, "knee_" + side: knee, "ankle_" + side: ankle,
                       "toe_" + side: ankle + [0, -.035, .14],
                       "shoulder_" + side: np.array([sign * .24, .53, 0.]),
                       "elbow_" + side: np.array([sign * .27, .27, .18]),
                       "hand_" + side: np.array([sign * .28, .12, .48])})
    return result


def render_video(trial, directory):
    """Render invented movement and a walker prop; neither is estimated from video."""
    raw = directory / "render.avi"
    width, height = 720, 480
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), trial.frame_rate, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic video")
    yaw = .85
    def point(p):
        x = np.cos(yaw) * p[0] + np.sin(yaw) * p[2]
        depth = -np.sin(yaw) * p[0] + np.cos(yaw) * p[2]
        return int(width * .46 + 175 * x), int(height * .42 - 175 * p[1] + 20 * depth)
    try:
        for frame, values in enumerate(trial.values):
            image = np.full((height, width, 3), (233, 238, 238), np.uint8)
            cv2.rectangle(image, (0, 350), (width, height), (218, 226, 225), -1)
            for x in range(-4, 6):
                cv2.line(image, point([x * .3, -1, -1]), point([x * .3, -1, 3]), (201, 213, 211), 1, cv2.LINE_AA)
            for z in np.arange(-1, 3, .3):
                cv2.line(image, point([-1.5, -1, z]), point([1.5, -1, z]), (201, 213, 211), 1, cv2.LINE_AA)
            body = joints(values)
            segments = [("pelvis", "chest"), ("chest", "head")]
            for side in ("l", "r"):
                segments += [("pelvis", "hip_" + side), ("hip_" + side, "knee_" + side),
                    ("knee_" + side, "ankle_" + side), ("ankle_" + side, "toe_" + side),
                    ("chest", "shoulder_" + side), ("shoulder_" + side, "elbow_" + side),
                    ("elbow_" + side, "hand_" + side)]
            for a, b in segments:
                color = (137, 136, 92) if b.endswith("_r") else (115, 121, 52)
                cv2.line(image, point(body[a]), point(body[b]), color, 15, cv2.LINE_AA)
            cv2.circle(image, point(body["head"]), 23, (128, 138, 99), -1, cv2.LINE_AA)
            for name in ("knee_l", "knee_r", "ankle_l", "ankle_r"):
                cv2.circle(image, point(body[name]), 7, (227, 235, 221), -1, cv2.LINE_AA)
            # A stylized walker prop occludes parts of the rendered figure.
            for x in (-.31, .31):
                cv2.line(image, point([x, .13, .48]), point([x, -.98, .6]), (100, 110, 113), 6, cv2.LINE_AA)
                cv2.line(image, point([x, -.98, .6]), point([x, -.98, -.1]), (100, 110, 113), 5, cv2.LINE_AA)
                cv2.circle(image, point([x, -.98, .6]), 8, (65, 73, 74), -1, cv2.LINE_AA)
            cv2.line(image, point([-.31, .13, .48]), point([.31, .13, .48]), (100, 110, 113), 6, cv2.LINE_AA)
            cv2.putText(image, "SYNTHETIC / NOT A PERSON", (22, 32), cv2.FONT_HERSHEY_SIMPLEX, .53, (83, 101, 94), 1, cv2.LINE_AA)
            cv2.putText(image, f"FRAME {frame:03d}    {trial.time[frame]:.2f}s", (22, height - 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (83, 101, 94), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
    video = directory / "synthetic.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "25", "-movflags", "+faststart", str(video)], check=True)
    return video.read_bytes()


def build(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trial = synthetic_trial()
    # Keep generated intermediates inside this repository, never in patient folders.
    cache = ROOT / ".pytest_cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ui-demo-", dir=cache) as temp:
        video = render_video(trial, Path(temp))
    template = Path(__file__).with_name("workspace.html").read_text(encoding="utf-8")
    payload = dict(trial=trial.to_dict(), video="data:video/mp4;base64," + base64.b64encode(video).decode("ascii"),
                   report=render_report(trial, planes={n: "sagittal" for n in trial.coordinate_names}, task="synthetic demonstration"))
    embedded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    output.write_text(template.replace("/*__DEMO_PAYLOAD__*/ null", embedded), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("demo.html"))
    args = parser.parse_args()
    print(build(args.output).resolve())


if __name__ == "__main__":
    main()
