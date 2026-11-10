"""Agreement statistics, including the worked example from Bland & Altman (2007)."""
import numpy as np
import pytest

from scipy import stats

from benchmarks.agreement import (bland_altman, bland_altman_repeated, curve_agreement, event_agreement,
                                  loa_from_anova)

# Table 1 of Bland & Altman (2007): cardiac ejection fraction by RV and IC, 12 subjects.
TABLE_1 = {
    1: [(7.83, 6.57), (7.42, 5.62), (7.89, 6.90), (7.12, 6.57), (7.88, 6.35)],
    2: [(6.16, 4.06), (7.26, 4.29), (6.71, 4.26), (6.54, 4.09)],
    3: [(4.75, 4.71), (5.24, 5.50), (4.86, 5.08), (4.78, 5.02), (6.05, 6.01), (5.42, 5.67)],
    4: [(4.21, 4.14), (3.61, 4.20), (3.72, 4.61), (3.87, 4.68), (3.92, 5.04)],
    5: [(3.13, 3.03), (2.98, 2.86), (2.85, 2.77), (3.17, 2.46), (3.09, 2.32)],
    6: [(3.12, 2.43), (5.92, 5.90), (6.42, 5.81), (5.92, 5.70)],
    7: [(6.27, 5.76), (7.13, 5.09), (6.62, 4.63), (6.58, 4.61)],
    8: [(6.93, 5.09), (4.54, 4.72), (4.81, 4.61), (5.11, 4.36), (5.29, 4.20), (5.39, 4.36), (5.57, 4.20)],
    9: [(4.48, 3.17), (4.92, 3.12), (3.97, 2.96)],
    10: [(4.22, 4.35), (4.65, 4.62), (4.74, 3.16), (4.44, 3.53), (4.50, 3.53)],
    11: [(6.78, 7.20), (6.07, 6.09), (6.52, 7.00), (6.42, 7.10), (6.41, 7.40), (5.76, 6.80)],
    12: [(5.06, 4.50), (4.72, 4.20), (4.90, 3.80), (4.80, 3.80), (4.90, 4.20), (5.10, 4.50)],
}


def test_limits_step_reproduces_the_paper_from_its_printed_values():
    # Paper text: MS subjects 4.2090856, residual 0.170714026, n = 12, sum m_i = 60, sum m_i^2 = 312,
    # divisor 4.9818182, heterogeneity 0.81062203, SD 0.99062408, limits -1.3394565 to 2.5437899.
    counts = [7, 3, 6, 4, 6, 4, 5, 5, 5, 5, 5, 5]  # any counts with these two sums give the same divisor
    assert sum(counts) == 60 and sum(c * c for c in counts) == 312
    result = loa_from_anova(0.6021667, 4.2090856, 0.170714026, counts)
    assert result["divisor"] == pytest.approx(4.9818182, abs=1e-7)
    assert result["between_variance"] == pytest.approx(0.81062203, abs=1e-7)
    assert result["sd"] == pytest.approx(0.99062408, abs=1e-7)
    assert result["loa"] == pytest.approx([-1.3394565, 2.5437899], abs=1e-6)


def test_repeated_measures_anova_on_the_published_table():
    differences = {s: [rv - ic for rv, ic in pairs] for s, pairs in TABLE_1.items()}
    result = bland_altman_repeated(differences)
    assert result["n_pairs"] == 60 and result["subjects"] == 12
    assert result["bias"] == pytest.approx(0.6021667, abs=1e-6)  # paper's mean difference
    values = [np.array(v) for v in differences.values()]
    everything = np.concatenate(values)
    # Total SS matches the paper's Table 2 (54.4942149). The paper's subject/residual split
    # (4.2090856 / 0.170714) is not reproduced by Table 1 as printed, and its text gives
    # sum m_i^2 = 312 where the table gives 314, so check the ANOVA against scipy instead.
    assert ((everything - everything.mean()) ** 2).sum() == pytest.approx(54.4942149, abs=1e-5)
    assert result["ms_subjects"] / result["ms_residual"] == pytest.approx(stats.f_oneway(*values).statistic)
    m = np.array([len(v) for v in values])
    assert result["divisor"] == pytest.approx((m.sum() ** 2 - (m ** 2).sum()) / (11 * m.sum()))


def test_negative_heterogeneity_is_clamped_and_reported():
    result = bland_altman_repeated([[0.0, 2.0], [1.0, 1.0]])
    assert result["between_variance"] == 0.0
    assert result["warnings"]


def test_repeated_needs_two_subjects():
    assert bland_altman_repeated([[1.0, 2.0]])["loa"] is None


def test_single_pair_limits_match_hand_calculation():
    result = bland_altman([11, 12, 13, 14], [10, 10, 10, 10])
    sd = np.std([1, 2, 3, 4], ddof=1)
    assert result["bias"] == 2.5
    assert result["loa"] == pytest.approx([2.5 - 1.96 * sd, 2.5 + 1.96 * sd])
    assert result["loa_ci"][0][0] < result["loa"][0] < result["loa_ci"][0][1]


def test_missing_pairs_are_excluded_and_counted():
    result = bland_altman([1, 2, np.nan, 4], [0, 0, 0, np.nan])
    assert result["n_pairs"] == 2 and result["excluded_pairs"] == 2 and result["loa"] is None


def test_event_matching_counts_misses_and_extras():
    result = event_agreement([1.01, 1.98, 3.5, 4.0], [1.0, 2.0, 3.0], tolerance_s=0.1)
    assert result["matched"] == 2 and result["missed"] == 1 and result["extra"] == 2
    assert result["errors_s"] == pytest.approx([0.01, -0.02])


def test_event_matching_uses_each_event_once():
    result = event_agreement([1.0], [0.98, 1.01], tolerance_s=0.1)
    assert result["matched"] == 1 and result["missed"] == 1 and result["errors_s"] == pytest.approx([-0.01])


def test_curve_agreement_separates_offset_from_shape():
    base = np.sin(np.linspace(0, 2 * np.pi, 101))
    result = curve_agreement([base + 3, base + 3], [base])
    assert result["offset"] == pytest.approx(3)
    assert result["rmse"] == pytest.approx(3)
    assert result["rmse_offset_removed"] == pytest.approx(0, abs=1e-12)
    assert result["pearson_r"] == pytest.approx(1)
