"""Static SVG/HTML report: no scripts, external assets or network requests."""
from html import escape
from pathlib import Path

import numpy as np

from medpose.errors import ErrorTable, reliability
from medpose.clinical.patterns import pattern_prompts
from medpose.clinical.gait import analyse_gait

DISCLAIMER = "Research use only, not clinically validated."
ERROR_CAVEAT = ("Error estimates come from benchmark populations and simulated occlusion, not this person's body, "
                "so the true error is unknown and could be larger.")


def _curve(time, values):
    valid = np.isfinite(values)
    if not valid.any():
        return '<p>No finite observations.</p>'
    lo, hi = float(values[valid].min()), float(values[valid].max())
    span = hi - lo or 1.
    duration = time[-1] - time[0] if len(time) > 1 else 1.
    segments, current = [], []
    for t, v in zip(time, values):
        if not np.isfinite(v):
            if current:
                segments.append(current)
            current = []
        else:
            current.append((35 + 530 * (t - time[0]) / duration, 140 - 110 * (v - lo) / span))
    if current:
        segments.append(current)
    strokes = []
    for segment in segments:
        points = ' '.join(f'{x:.3f},{y:.3f}' for x, y in segment)
        strokes.append(f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="2"/>')
        if len(segment) == 1:
            x, y = segment[0]
            strokes.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2"/>')
    return (f'<svg viewBox="0 0 600 175" role="img" aria-label="Coordinate over elapsed time">'
            f'<text x="2" y="20">{hi:g}</text><text x="2" y="155">{lo:g}</text>'
            + ''.join(strokes) + f'<text x="35" y="172">{time[0]:g} s</text>'
            f'<text x="520" y="172">{time[-1]:g} s</text></svg>')


def render_report(trial, *, planes, task, errors=None, gait_config=None, rules=None):
    """Return standalone HTML. Caller supplies coordinate planes; none are guessed."""
    if any(n not in planes for n in trial.coordinate_names):
        raise ValueError("Provide a plane for every coordinate")
    errors = ErrorTable.load() if errors is None else errors
    sections = []
    for i, name in enumerate(trial.coordinate_names):
        error = errors.lookup(trial.recording_mode, name, planes[name], task)
        check = reliability(error)
        values = trial.values[:, i]
        finite = values[np.isfinite(values)]
        summary = f"Range {np.ptp(finite):g} {trial.units[i]}; mean {finite.mean():g} {trial.units[i]}" if len(finite) else "No finite observations"
        missing = int(np.isnan(values).sum())
        confidence = "Confidence absent"
        if trial.confidence is not None:
            known = trial.confidence[:, i][np.isfinite(trial.confidence[:, i])]
            confidence = f"Confidence minimum {known.min():g}" if len(known) else "Confidence unknown"
        css = "" if check.status == "reliable" else "uncertain"
        sections.append(f'<section class="{css}"><h2>{escape(name)} ({trial.units[i]})</h2>'
                        f'<p>{escape(summary)}; missing frames {missing}. {confidence}.</p>'
                        f'<p><strong>{escape(check.status)}</strong>: {escape(error.label)}. '
                        f'{escape(check.threshold_status)}.</p>{_curve(trial.time, values)}</section>')
    prompts = pattern_prompts(trial, errors=errors, rules=rules)
    prompt_html = ''.join(f'<li>{escape(p["prompt"] or p["name"] + ": " + p["status"])}</li>' for p in prompts)
    gait_html = '<p>Gait analysis not requested; no contact thresholds assumed.</p>'
    if gait_config is not None:
        gait = analyse_gait(trial, gait_config, planes=planes, task=task, errors=errors)
        gait_html = '<p>Provisional contact candidates; timing error not yet measured.</p>'
        for side, data in gait["sides"].items():
            gait_html += f'<h3>{side.capitalize()}</h3><p>{escape(data["events"]["reason"])}</p><ul>'
            for label in ("duration_s", "stance_percent", "cadence_steps_min"):
                summary = data[label]
                gait_html += f'<li>{label}: mean {summary["mean"]}; sample SD {summary["sd"]}; cycles {summary["count"]}; error not yet measured</li>'
            gait_html += '</ul>'
    supports = ', '.join(trial.supports) or "None reported"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>MedPose research report</title><style>
body{{font:16px system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#17242e;background:#fff}}
header{{border-bottom:3px solid #245d75}}.notice{{background:#fff3d4;padding:1rem}}
section{{border-top:1px solid #ccd4d8;padding:1rem 0;break-inside:avoid}}.uncertain{{color:#666}}
svg{{width:100%;max-height:240px}}svg text{{fill:currentColor;font-size:12px}}
</style></head><body><header><h1>MedPose movement review</h1><p class="notice">{DISCLAIMER}<br>{ERROR_CAVEAT}</p>
<p>Mode: {escape(trial.recording_mode)} · Cameras: {trial.camera_count} · Frame rate: {trial.frame_rate:g} fps</p>
<p>Supports worn/used: {escape(supports)}</p><p>Engine: {escape(trial.engine_name)} {escape(trial.engine_version)} · Task: {escape(task)}</p>
<p>Grey values remain visible because reliability is unknown or outside the configured limit. No diagnosis or treatment advice.</p></header>
{''.join(sections)}<section><h2>Gait summaries</h2>{gait_html}</section>
<section><h2>Discussion prompts</h2><ul>{prompt_html}</ul></section>
<footer>Self-contained offline report. No uploads. No forces or kinetics estimated.</footer></body></html>'''


def save_report(trial, path, **kwargs):
    Path(path).write_text(render_report(trial, **kwargs), encoding="utf-8")
