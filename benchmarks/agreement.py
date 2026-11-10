"""Agreement statistics that gait clinicians and journals expect.

* Bland-Altman limits of agreement for one pair per subject: Bland & Altman
  (1986), Lancet 327:307-310. Approximate CI of each limit uses
  SE = sqrt(3 s^2 / n) from the same paper.
* Limits of agreement with several pairs per subject, true value varying
  (e.g. one pair per gait cycle): Bland & Altman (2007), J Biopharm Stat
  17:571-582, doi:10.1080/10543400701329422, section "Method where the true value
  varies". Note that paper corrects the divisor given in their 1999 paper.
* Event timing agreement: one-to-one matching of predicted to reference events.

Markers are a reference, not the truth: soft tissue moves and marker placement
varies between assessors, so report "agreement with marker-based reference".
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import stats

BA_1986 = "Bland & Altman 1986, Lancet 327:307-310"
BA_2007 = "Bland & Altman 2007, J Biopharm Stat 17:571-582, doi:10.1080/10543400701329422"


def _pairs(prediction, reference):
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference need one value per pair")
    if np.isinf(prediction).any() or np.isinf(reference).any():
        raise ValueError("Use NaN, not infinity, for missing values")
    valid = np.isfinite(prediction) & np.isfinite(reference)
    return prediction[valid] - reference[valid], int((~valid).sum())


def bland_altman(prediction, reference, *, ci=0.95):
    """Bias and 95% limits of agreement for independent pairs (one per subject).

    Differences are prediction minus reference. Pairs with a missing value are
    excluded and counted. Fewer than 3 pairs gives no limits.
    """
    differences, excluded = _pairs(prediction, reference)
    n = len(differences)
    result = dict(method="independent pairs", origin=BA_1986, n_pairs=n, excluded_pairs=excluded,
                  bias=None, sd=None, loa=None, bias_ci=None, loa_ci=None)
    if n < 3:
        result["status"] = "fewer than 3 complete pairs"
        return result
    bias, sd = float(differences.mean()), float(differences.std(ddof=1))
    t = float(stats.t.ppf(0.5 + ci / 2, n - 1))
    se_bias, se_limit = sd / np.sqrt(n), np.sqrt(3 * sd ** 2 / n)
    limits = [bias - 1.96 * sd, bias + 1.96 * sd]
    result.update(status="computed", bias=bias, sd=sd, loa=limits,
                  bias_ci=[bias - t * se_bias, bias + t * se_bias],
                  loa_ci=[[x - t * se_limit, x + t * se_limit] for x in limits])
    return result


def bland_altman_repeated(differences_by_subject):
    """Limits of agreement when each subject contributes several pairs.

    ``differences_by_subject`` maps an anonymous key (or a sequence position) to
    that subject's prediction-minus-reference differences, e.g. one per gait cycle.
    Implements the one-way ANOVA estimator of Bland & Altman (2007) for a true
    value that varies: within-subject variance is the residual mean square; the
    between-subject (heterogeneity) component is (MS_subjects - MS_residual) / D
    with D = ((sum m_i)^2 - sum m_i^2) / ((n - 1) sum m_i). A negative heterogeneity
    estimate is set to zero and reported. The bias is the mean of all differences.
    Assumes repeated differences within a subject are independent.
    """
    groups = list(differences_by_subject.values() if isinstance(differences_by_subject, Mapping)
                  else differences_by_subject)
    cleaned, excluded = [], 0
    for group in groups:
        values = np.asarray(group, dtype=float).reshape(-1)
        if np.isinf(values).any():
            raise ValueError("Use NaN, not infinity, for missing differences")
        excluded += int(np.isnan(values).sum())
        values = values[np.isfinite(values)]
        if len(values):
            cleaned.append(values)
    result = dict(method="repeated pairs, true value varies", origin=BA_2007, subjects=len(cleaned),
                  n_pairs=int(sum(len(g) for g in cleaned)), excluded_pairs=excluded, bias=None,
                  within_variance=None, between_variance=None, sd=None, loa=None, warnings=[])
    n = len(cleaned)
    total = sum(len(g) for g in cleaned)
    if n < 2 or total - n < 1:
        result["status"] = "need at least 2 subjects and one repeated pair"
        return result
    everything = np.concatenate(cleaned)
    grand = everything.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in cleaned)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in cleaned)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (total - n)
    limits = loa_from_anova(grand, ms_between, ms_within, [len(g) for g in cleaned])
    if limits.pop("clamped"):
        result["warnings"].append("Heterogeneity estimate was negative and has been set to zero")
    result.update(status="computed", ms_subjects=float(ms_between), ms_residual=float(ms_within), **limits)
    return result


def loa_from_anova(bias, ms_subjects, ms_residual, counts):
    """Limits of agreement from one-way ANOVA mean squares (Bland & Altman 2007).

    ``counts`` are the numbers of pairs per subject (m_i). Separate from the ANOVA
    so the step can be checked against the paper's printed intermediate values.
    """
    m = np.asarray(counts, dtype=float)
    n = len(m)
    divisor = (m.sum() ** 2 - (m ** 2).sum()) / ((n - 1) * m.sum())
    between = (ms_subjects - ms_residual) / divisor
    clamped = between < 0
    between = max(between, 0.0)
    sd = float(np.sqrt(ms_residual + between))
    return dict(bias=float(bias), divisor=float(divisor), within_variance=float(ms_residual),
                between_variance=float(between), sd=sd, clamped=bool(clamped),
                loa=[float(bias - 1.96 * sd), float(bias + 1.96 * sd)])


def event_agreement(predicted_s, reference_s, *, tolerance_s):
    """Match each reference event to the nearest unused predicted event within tolerance.

    Returns signed errors (predicted minus reference, seconds), counts of missed
    reference events and extra predicted events. Matching is greedy by smallest
    absolute difference, so each event is used at most once.
    """
    predicted = np.sort(np.asarray(predicted_s, dtype=float).reshape(-1))
    reference = np.sort(np.asarray(reference_s, dtype=float).reshape(-1))
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Event times must be finite")
    if not np.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    candidates = sorted((abs(p - r), i, j) for i, r in enumerate(reference) for j, p in enumerate(predicted)
                        if abs(p - r) <= tolerance_s)
    used_ref, used_pred, matches = set(), set(), []
    for _, i, j in candidates:
        if i in used_ref or j in used_pred:
            continue
        used_ref.add(i); used_pred.add(j)
        matches.append((float(reference[i]), float(predicted[j] - reference[i])))
    matches.sort()
    errors = np.array([e for _, e in matches])
    return dict(tolerance_s=float(tolerance_s), matched=len(matches),
                missed=len(reference) - len(matches), extra=len(predicted) - len(matches),
                errors_s=errors.tolist(), mean_error_s=float(errors.mean()) if len(errors) else None,
                mean_absolute_error_s=float(np.abs(errors).mean()) if len(errors) else None,
                max_absolute_error_s=float(np.abs(errors).max()) if len(errors) else None)


def curve_agreement(prediction_curves, reference_curves):
    """Compare gait-cycle-normalised curves (cycles, samples) from the two systems.

    Uses the mean curve of each system, so cycle counts may differ. Returns RMSE,
    RMSE after removing the mean offset, the offset itself, and Pearson r of the
    mean curves (shape agreement independent of offset).
    """
    def mean_curve(curves):
        rows = [np.asarray(c, dtype=float) for c in curves if c is not None]
        if not rows:
            return None
        array = np.vstack(rows)
        if not np.isfinite(array).all():
            raise ValueError("Normalised curves must be complete")
        return array.mean(axis=0)
    a, b = mean_curve(prediction_curves), mean_curve(reference_curves)
    if a is None or b is None:
        return dict(status="no complete cycles", rmse=None, offset=None, rmse_offset_removed=None, pearson_r=None)
    if a.shape != b.shape:
        raise ValueError("Curves must use the same number of samples")
    difference = a - b
    offset = float(difference.mean())
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return dict(status="computed", rmse=float(np.sqrt(np.mean(difference ** 2))), offset=offset,
                rmse_offset_removed=float(np.sqrt(np.mean((difference - offset) ** 2))), pearson_r=r,
                prediction_cycles=sum(c is not None for c in prediction_curves),
                reference_cycles=sum(c is not None for c in reference_curves))
