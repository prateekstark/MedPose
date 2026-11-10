"""Gait report in the layout gait clinicians read: cycle graphs plus a parameter table.

Static HTML/SVG, no scripts or network requests. Left is drawn red and right blue
(and right dashed, so the sides differ without colour). An optional normative band
(mean +/- 1 SD) and an optional previous session (thin line) can be overlaid.
"""
from __future__ import annotations

from html import escape

import numpy as np

from medpose.clinical.parameters import mean_curves
from medpose.report.html import DISCLAIMER, ERROR_CAVEAT

LEFT, RIGHT, PREVIOUS, NORM = "#c0392b", "#1f6fb2", "#6b7780", "#d5dbe0"
GRAPHS = (("pelvis_tilt", "Pelvic tilt"), ("hip_flexion", "Hip flexion (+) / extension"),
          ("knee_angle", "Knee flexion"), ("ankle_angle", "Ankle dorsiflexion (+) / plantarflexion"),
          ("hip_adduction", "Hip adduction (+) / abduction"), ("hip_rotation", "Hip internal (+) / external rotation"))
LOW_TRUST = {"hip_adduction", "hip_rotation", "pelvis_tilt"}


def _fmt(value, digits=1):
    if value is None:
        return "–"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text  # no "-0.0"


def _summary_cell(summary):
    if summary.get("mean") is None:
        return "–"
    sd = f" ± {_fmt(summary['sd'])}" if summary.get("sd") is not None else ""
    return f"{_fmt(summary['mean'])}{sd} <small>(n={summary['count']})</small>"


def _graph(title, series, *, normative=None, toe_off=None):
    """series: list of (label, curve, colour, dashed, sd_curve or None, stroke width)."""
    curves = [np.asarray(c) for _, c, _, _, _, _ in series if c is not None]
    if not curves:
        return f'<figure><figcaption>{escape(title)}</figcaption><p class="muted">Not available for these cycles.</p></figure>'
    values = np.concatenate(curves + ([np.asarray(normative[0]) - np.asarray(normative[1]),
                                       np.asarray(normative[0]) + np.asarray(normative[1])] if normative else []))
    lo, hi = float(np.floor(values.min() / 5) * 5), float(np.ceil(values.max() / 5) * 5)
    if hi - lo < 10:
        hi, lo = hi + 5, lo - 5
    width, height, left, top, plot_w, plot_h = 320, 200, 36, 10, 270, 160
    x = lambda p: left + plot_w * p / 100
    y = lambda v: top + plot_h * (hi - v) / (hi - lo)
    def path(curve):
        n = len(curve)
        return " ".join(f"{x(100 * i / (n - 1)):.1f},{y(v):.1f}" for i, v in enumerate(curve))
    parts = [f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#9aa5ad"/>']
    if lo < 0 < hi:
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#9aa5ad" stroke-dasharray="2 3"/>')
    if normative:
        mean, sd = np.asarray(normative[0]), np.asarray(normative[1])
        upper, lower = mean + sd, (mean - sd)[::-1]
        n = len(mean)
        pts = [f"{x(100 * i / (n - 1)):.1f},{y(v):.1f}" for i, v in enumerate(upper)]
        pts += [f"{x(100 * (n - 1 - i) / (n - 1)):.1f},{y(v):.1f}" for i, v in enumerate(lower)]
        parts.append(f'<polygon points="{" ".join(pts)}" fill="{NORM}" stroke="none"/>')
    for label, curve, colour, dashed, sd_curve, stroke in series:
        if curve is None:
            continue
        if sd_curve is not None:
            c, s = np.asarray(curve), np.asarray(sd_curve)
            n = len(c)
            pts = [f"{x(100 * i / (n - 1)):.1f},{y(v):.1f}" for i, v in enumerate(c + s)]
            pts += [f"{x(100 * (n - 1 - i) / (n - 1)):.1f},{y(v):.1f}" for i, v in enumerate((c - s)[::-1])]
            parts.append(f'<polygon points="{" ".join(pts)}" fill="{colour}" fill-opacity="0.12" stroke="none"/>')
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(f'<polyline points="{path(curve)}" fill="none" stroke="{colour}" stroke-width="{stroke}"{dash}>'
                     f'<title>{escape(label)}</title></polyline>')
    for side, percent, colour in toe_off or ():
        if percent is not None:
            parts.append(f'<line x1="{x(percent):.1f}" x2="{x(percent):.1f}" y1="{top}" y2="{top + plot_h}" '
                         f'stroke="{colour}" stroke-width="1" stroke-dasharray="1 3"><title>{side} toe-off</title></line>')
    parts.append(f'<text x="{left - 4}" y="{top + 4}" text-anchor="end">{hi:g}</text>'
                 f'<text x="{left - 4}" y="{top + plot_h}" text-anchor="end">{lo:g}</text>'
                 f'<text x="{left}" y="{height - 12}">0</text><text x="{left + plot_w}" y="{height - 12}" text-anchor="end">100% cycle</text>')
    return (f'<figure><figcaption>{escape(title)} (deg)</figcaption><svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{escape(title)} over the gait cycle">{"".join(parts)}</svg></figure>')


def _sd_curves(gait, variable):
    out = {}
    for side in ("left", "right"):
        coords = gait["sides"][side]["coordinates"]
        name = f"{variable}_{side[0]}" if f"{variable}_{side[0]}" in coords else variable
        rows = [c for c in coords.get(name, {}).get("normalized", []) if c is not None]
        out[side] = np.std(rows, axis=0, ddof=1).tolist() if len(rows) > 1 else None
    return out


def render_gait_report(trial, gait, parameters, *, task, events_method, gps=None, normative=None,
                       previous_gait=None, previous_parameters=None, previous_label="previous session"):
    """Standalone HTML for one walking trial (optionally against a previous one)."""
    variables = [v for v, _ in GRAPHS]
    current = mean_curves(gait, variables)
    previous = mean_curves(previous_gait, variables) if previous_gait is not None else None
    stance = {s: gait["sides"][s]["stance_percent"]["mean"] for s in ("left", "right")}
    figures = []
    for variable, title in GRAPHS:
        sd = _sd_curves(gait, variable)
        series = [("Left", current["left"][variable], LEFT, False, sd["left"], 2.2),
                  ("Right", current["right"][variable], RIGHT, True, sd["right"], 2.2)]
        if previous is not None:
            series += [(f"Left, {previous_label}", previous["left"][variable], LEFT, False, None, 0.9),
                       (f"Right, {previous_label}", previous["right"][variable], RIGHT, True, None, 0.9)]
        band = None
        if normative and variable in normative["variables"] and normative["variables"][variable]["sd"]:
            band = (normative["variables"][variable]["mean"], normative["variables"][variable]["sd"])
        name = title + (" — low trust" if variable in LOW_TRUST else "")
        figures.append(_graph(name, series, normative=band,
                              toe_off=[("Left", stance["left"], LEFT), ("Right", stance["right"], RIGHT)]))

    temporal_rows = []
    for key, label, digits in (("duration_s", "Cycle duration (s)", 2), ("stance_percent", "Stance (% cycle)", 1),
                               ("cadence_steps_min", "Cadence estimate (steps/min)", 0)):
        cells = [_summary_cell(gait["sides"][s][key]) for s in ("left", "right")]
        temporal_rows.append(f"<tr><th>{label}</th><td>{cells[0]}</td><td>{cells[1]}</td><td>–</td></tr>")
    if gait["speed"].get("status") == "signed progression speed":
        cells = [_summary_cell(gait["speed"]["sides"][s]) for s in ("left", "right")]
        temporal_rows.append(f"<tr><th>Walking speed (m/s)</th><td>{cells[0]}</td><td>{cells[1]}</td><td>–</td></tr>")

    param_rows = []
    for name, entry in parameters["parameters"].items():
        left, right = entry["sides"]["left"], entry["sides"]["right"]
        prev = ""
        if previous_parameters is not None:
            p = previous_parameters["parameters"].get(name)
            prev = (f"<td>{_fmt(p['sides']['left']['mean'])} / {_fmt(p['sides']['right']['mean'])}</td>"
                    if p else "<td>–</td>")
        label = left.get("error_label") or right.get("error_label") or "error not yet measured"
        param_rows.append(f'<tr><th>{escape(entry["description"])}</th><td>{_summary_cell(left)}</td>'
                          f'<td>{_summary_cell(right)}</td><td>{_fmt(entry["left_minus_right"])}</td>{prev}'
                          f'<td class="muted">{escape(label)}</td></tr>')
    prev_head = f"<th>{escape(previous_label)} (L / R)</th>" if previous_parameters is not None else ""

    gps_html = ""
    if gps is not None:
        rows = "".join(f"<tr><th>{escape(v)}</th><td>{_fmt(gps['sides']['left']['variable_scores'][v])}</td>"
                       f"<td>{_fmt(gps['sides']['right']['variable_scores'][v])}</td></tr>" for v in gps["variables"])
        gps_html = (f'<section><h2>Deviation from reference curves</h2><p class="muted">{escape(gps["caveat"])}. '
                    f'Reference: {escape(gps["normative_source"])} ({gps["normative_people"]} people).</p>'
                    f'<table><tr><th>Variable (RMS deviation, deg)</th><th>Left</th><th>Right</th></tr>{rows}'
                    f'<tr><th>Score</th><td>{_fmt(gps["sides"]["left"]["score"])}</td>'
                    f'<td>{_fmt(gps["sides"]["right"]["score"])}</td></tr></table></section>')

    cycles = {s: len(gait["sides"][s]["events"]["cycles"]) for s in ("left", "right")}
    supports = ", ".join(trial.supports) or "none recorded"
    change_note = ""
    if previous_gait is not None:
        change_note = ('<p class="notice">Differences from the previous session are descriptive. Whether a change '
                       'is larger than measurement noise needs the person’s own repeat sessions in the same setup '
                       '(minimal detectable change; see medpose.clinical.change).</p>')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>MedPose gait report</title><style>
body{{font:15px system-ui,sans-serif;max-width:1040px;margin:1.5rem auto;padding:0 1rem;color:#17242e;background:#fff}}
header{{border-bottom:3px solid #245d75}}.notice{{background:#fff3d4;padding:.75rem 1rem}}.muted{{color:#5b6770}}
.graphs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:.5rem 1rem}}
figure{{margin:0}}figcaption{{font-weight:600;font-size:14px}}svg{{width:100%;height:auto}}svg text{{fill:#5b6770;font-size:10px}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0}}th,td{{border-bottom:1px solid #dde3e7;padding:.35rem .5rem;text-align:left;vertical-align:top}}
td small{{color:#5b6770}}.key span{{display:inline-block;margin-right:1rem}}
@media print{{.graphs{{grid-template-columns:repeat(2,1fr)}}section{{break-inside:avoid}}}}
</style></head><body><header><h1>Gait report</h1>
<p class="notice">{DISCLAIMER} {ERROR_CAVEAT} Results are prompts for discussion with the clinicians, not a diagnosis.</p>
<p>Task: {escape(task)} · Mode: {escape(trial.recording_mode)} ({trial.camera_count} camera(s), {trial.frame_rate:g} fps) ·
Engine: {escape(trial.engine_name)} {escape(trial.engine_version)}</p>
<p>Supports: {escape(supports)} · Cycles analysed: left {cycles["left"]}, right {cycles["right"]}</p>
<p class="muted">Gait events: {escape(events_method)}. Angles use the engine's model conventions; absolute ankle and pelvis
values are not interchangeable with a gait lab's. Frontal and transverse graphs are marked low trust.</p></header>
{change_note}
<section><h2>Joint angles over the gait cycle</h2>
<p class="key"><span style="color:{LEFT}">━ Left (mean ± SD)</span><span style="color:{RIGHT}">╍ Right (mean ± SD)</span>
{"<span style='color:#6b7780'>thin: " + escape(previous_label) + "</span>" if previous_gait is not None else ""}
{"<span>grey band: reference mean ± 1 SD</span>" if normative else ""}<span>dotted: toe-off</span></p>
<div class="graphs">{"".join(figures)}</div></section>
<section><h2>Timing</h2><table><tr><th></th><th>Left</th><th>Right</th><th>L − R</th></tr>{"".join(temporal_rows)}</table>
<p class="muted">Cadence assumes alternating steps (120 / cycle duration).</p></section>
<section><h2>Gait parameters (degrees)</h2><table><tr><th>Parameter</th><th>Left mean ± SD</th><th>Right mean ± SD</th>
<th>L − R</th>{prev_head}<th>Error</th></tr>{"".join(param_rows)}</table></section>
{gps_html}
<footer class="muted">Self-contained offline report. No uploads. No forces estimated: the walker and belt carry
unmeasured load.</footer></body></html>'''
