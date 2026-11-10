"""Generated walking with known events and angles. Not a model of any person's gait.

Walking is along +x at 1 m/s with a 1.0 s gait cycle, sampled at 100 Hz. Relative to
the pelvis, the left heel peaks forward at t = 1, 2, 3... (heel strikes) and the
left toe is furthest back at t = 1.6, 2.6... (toe-offs), so stance is 60%. The right
side is the same, half a cycle later. Knee angle ramps 0 -> 60 deg linearly through
each left cycle, so parameters have exact expected values.
"""
import numpy as np

from medpose.feet import FootPositions
from medpose.trial import Trial

RATE = 100.0
PERIOD = 1.0
STANCE = 0.6


def walking_trial(seconds=5.2, rate=RATE, knee_offset=0.0, supports=("posterior walker",)):
    time = np.arange(int(round(seconds * rate)) + 1) / rate
    pelvis = np.column_stack([time, np.full_like(time, 0.9), np.zeros_like(time)])
    feet = {}
    for side, shift in (("left", 0.0), ("right", 0.5 * PERIOD)):
        phase = 2 * np.pi * (time - shift) / PERIOD
        heel_rel = 0.3 * np.cos(phase)
        # Minimum of cos at phase pi; shift so the toe minimum lands at STANCE of the cycle.
        toe_rel = 0.3 * np.cos(phase - 2 * np.pi * (STANCE - 0.5)) + 0.2
        points = np.zeros((len(time), 2, 3))
        points[:, 0, 0] = pelvis[:, 0] + heel_rel
        points[:, 1, 0] = pelvis[:, 0] + toe_rel
        points[:, :, 1] = 0.02 * (1 - np.cos(phase))[:, None]
        points[:, 0, 2] = points[:, 1, 2] = -0.1 if side == "left" else 0.1
        feet[side] = points
    fraction = np.mod(time, PERIOD) / PERIOD
    knee_l = 60 * fraction + knee_offset
    knee_r = 60 * np.mod(time - 0.5 * PERIOD, PERIOD) / PERIOD + knee_offset
    hip = 20 * np.cos(2 * np.pi * time / PERIOD)
    ankle = 10 * np.sin(2 * np.pi * time / PERIOD)
    names = ("pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "hip_flexion_l", "hip_flexion_r",
             "knee_angle_l", "knee_angle_r", "ankle_angle_l", "ankle_angle_r")
    values = np.column_stack([pelvis, np.zeros_like(time), hip, -hip, knee_l, knee_r, ankle, -ankle])
    foot_positions = FootPositions(feet["left"], feet["right"], [0, 1, 0], 0.0, "generated sole points")
    return Trial(time, names, values, "2_phone", "synthetic", "1", 2, rate, supports,
                 foot_positions=foot_positions)


def expected_left_strikes(seconds=5.2):
    return [float(t) for t in np.arange(1, seconds, PERIOD)]
