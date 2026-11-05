# Review of the external research output

Reviewed 2026-11-05 by Claude. This covers `fact_check.md`, `novelty_check.md` and `benchmarks/metrics.py`, produced by ChatGPT 6 Astra. Only the items listed as confirmed were checked independently; every other claim in those two documents remains the agent's unverified report.

## Confirmed at the primary source
| Claim | Evidence |
|---|---|
| PBL runtime on a 6 s, one-person clip: pose 10 s on A100 or 1 min on RTX 3050 Laptop; fitting 1 min or 4.5 min | Read directly in the PBL supplement, Runtime Experiments |
| PBL median absolute error: control 2.51°, neurological 3.32°, prosthesis 2.97°, pediatric 3.17° | Read directly in Supplementary Table 2 |
| PBL's clinical group included assistive devices, walkers, AFOs and therapist gait-belt assistance, which it says add error | Read directly in the supplement, Detailed Analysis |
| PBL's GDI uses hip flexion, hip adduction and knee flexion with 12 PCA components, so it isn't the original GDI | Read directly in the supplement |
| PBL code is AGPL-3.0, offers CPU install, documents one camera only, and defaults to Samsung S20 intrinsics | Project README |
| B3D reports medians of per-trial errors, with one offset per method and joint pooled across the cohort | B3D §3.6, §3.8, §4 |
| The Shriners cohort has 27 people, age 11.3±3.3, 12 with CP, some with crutches or walkers, 8 cameras at 60 Hz, Vicon reference | arXiv 2601.22412v2 §§2.2-2.4 |
| 55 typically developing children, 3-17 years, comfortable and ±30% treadmill speeds, per-child data files | PMC10126839 |

## Corrections and open conflicts
| Item | Status |
|---|---|
| B3D offset granularity | The paper says per (method, joint), not "joint group" as `fact_check.md` states |
| ChildGait contents | **Conflict.** The paper is reported to say the public data is pose sequences. The dataset card describes videos with pose sequences under tiered access. Check the file tree before relying on either |
| OpenCap Monocular filter values (6/4/8 Hz) | Not confirmed. Our check saw them only in a machine summary of Table S1, and the readable text doesn't mention filtering. Read the table before citing them |
| PBL Data Availability (MMMC not public) | Not confirmed by us: the journal page redirected to a sign-in flow. Treat MMMC as unavailable until told otherwise, and ask the authors |
| B3D 0.03-0.14 frames/s for PBL vs the PBL supplement's runtimes | Unresolved. B3D names no inference hardware, and the two measure different things. Measure it ourselves |
| B3D code and weights | No release found by either check; absence isn't proven |

## Consequences for comparisons
- **Don't compare different statistics.** PBL and B3D report medians. OpenCap reports means. `metrics.py` now has `median_absolute_error` for this reason.
- **Don't compare different reference systems.** MMMC is markerless. BioCV and MoVi are marker-based.
- **Don't compare different corrections.** `bias_corrected_mae` removes a per-trial offset, which flatters a method more than B3D's cohort-pooled offset. A cohort-pooled version needs B3D's exact estimator, which is still unverified.
- **Don't use raw `mae` on coordinates that wrap at ±180°,** such as pelvis rotation. Use `wrapped_difference` for those first.

## Novelty search limits
Scholar, MICCAI, TNSRE and Journal of Biomechanics coverage is incomplete by the agent's own account. The three broad claims shouldn't be presented as firsts. Repeat the search before any submission.
