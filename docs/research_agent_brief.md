# Research brief for an external agent

Written for any capable model with live web access (currently GPT6 Luna).

Paste everything below this line. It has no patient information, and none should be added.

---

You're helping on MedPose, an open research project that estimates 3D human movement from one or more ordinary cameras, offline. The focus is movement analysis for cerebral palsy (CP), including children who walk with a walker and have dystonic movement. It isn't a clinical product.

Your time may be limited. Work through the tasks in order. Tasks 1 and 2 matter most and don't depend on the code.

## Rules
1. **Every factual claim needs a direct source link** to the primary page: the paper, repo, licence file or dataset page. Secondary blogs and aggregators don't count as verification.
2. **Never guess.** Use these statuses exactly:
   - `VERIFIED`: you read it in the primary source.
   - `CONTRADICTED`: the primary source says otherwise. State what it says.
   - `NOT FOUND`: you looked and couldn't establish it. List where you looked.
3. For numbers, quote the exact value with its table or section number. Say whether it's bias-corrected, and which population and reference system it used.
4. Don't create accounts, accept licences, request dataset access or download gated data. Report what access requires instead.
5. If you can't access a page, say so. Don't reconstruct its contents from memory.
6. If you have repo access, work only on the branch `research-agent/research`, and add or change only the paths named in each task. Otherwise, return your files in chat.

## Task 1: fact-check sheet
Create `docs/research/fact_check.md` as a table with these columns: item, status, finding, source link, date checked.

**Biomechanical 3D Body (B3D), arXiv 2608.29928**
- Are code and weights released? Where, and under what licence?
- Confirm these median per-trial joint-angle errors, in degrees, from Tables S1/S2 or wherever they appear:

| Dataset | B3D | PBL | HSMR | OpenCap Monocular |
|---|---|---|---|---|
| BML-MoVi | 5.00 | 5.12 | 9.75 | 6.96 |
| MMMC | 5.75 | 4.75 | 9.74 | 11.87 |
| MMMC lower limb | 3.66 | 3.24 | 6.36 | 7.16 |
| BioCV lower limb | 4.04 | 4.10 | 6.11 | 5.39 |

- Does OpenCap Monocular reject 400 of 1,693 MMMC inputs? Are these numbers bias-corrected? How is the bias correction done?
- Confirm the reported speeds: about 15-20 frames/s for B3D and about 0.03-0.14 frames/s for PBL, with the hardware used.

**Portable Biomechanics Lab (PBL), arXiv 2507.08268 and any npj Digital Medicine version**
- Where's the code (project page: `intelligentsensingandrehabilitation.github.io/MonocularBiomechanics`)? What's its licence, what are its dependencies, and does it need a GPU, and how much VRAM?
- Can it take more than one camera?
- What are the reported errors by population (neurological, prosthesis, pediatric, control)?

**OpenCap Monocular, arXiv 2603.24733**
- Confirm 4.8° rotational and 3.4 cm translational error, and the per-activity values if given.
- Is the code licence PolyForm Noncommercial?

**Datasets.** For each one: what it contains, whether raw video is included, ages and populations, number of cameras and frame rate, how to get access, licence, and total download size.
- OpenCap validation dataset (SimTK).
- BioCV (University of Bath).
- BML-MoVi.
- Gillette CP gait data from Kidziński et al. 2020 (`github.com/stanfordnmbl/mobile-gaitlab`, `simtk.org/projects/video-gaitlab`). **Key question: is raw video available, or only OpenPose keypoints?** Also list the labels and any upper bound the paper gives on GDI accuracy.
- Children Gait Video (CGV), arXiv 2608.00371. Is there a working download link? What exactly is public?
- MMMC clinical cohort. Is it public? Is arXiv 2601.22412 the right citation?
- AddBiomechanics Dataset 1.0.

**Models.** For each one: licence for code and weights, whether weights are gated, and GPU memory needed for inference.
- SAM 3D Body, plus MHR.
- GVHMR.
- WHAM.
- HSMR.
- Sapiens.
- SAM 3.

**Other**
- Pediatric normative gait data that's publicly available, with raw curves, ages and speeds, including anything usable for speed-matched comparison.
- Exact definition and inputs needed to compute the Gait Deviation Index (Schwartz and Rozumalski, 2008), and whether the reference feature set is public.
- Licence for the OpenSim bone meshes used by the LaiUhlrich2022 model, and whether they may be redistributed.

## Task 2: novelty check
Create `docs/research/novelty_check.md`. We may write a paper on a single biomechanical fitting method that scales from 1 to 4 ordinary cameras, with these three claims. For each one, determine whether it has already been done.

1. **Accuracy vs number of cameras:** joint-angle error per joint and per anatomical plane, measured against marker-based mocap, for 1, 2, 3 and 4 views using the same method.
2. **Assistive-device occlusion:** markerless pose or biomechanics for people using walkers, crutches, gait trainers or a helper in frame, including simulated occlusion added to videos with ground truth.
3. **Fast involuntary movement:** whether monocular or multi-view pipelines preserve dystonia, chorea or tremor rather than smoothing them away, including any frequency-response evaluation.

For each claim, give:
- The 3-5 closest papers, with link, year, venue, population, method, and how close each one is to our claim.
- A verdict: `OPEN`, `PARTLY DONE` or `ALREADY DONE`, with your reasoning.
- What specifically would still be new.

Search arXiv, PubMed, Google Scholar and the CVPR, ICCV, ECCV, MICCAI, TNSRE and J Biomech proceedings, through the current month. List your search queries. **Prefer finding a paper that kills our idea over reassuring us.**

## Task 3: benchmark metrics module
Create `benchmarks/__init__.py`, `benchmarks/metrics.py` and `tests/test_metrics.py`. Use Python 3.9-compatible code with only numpy and the standard library, plus pytest for the tests.

Implement these on time-aligned arrays with shape `(frames, coordinates)`, in degrees. NaN marks missing frames and must be excluded, and the count of excluded frames must be reported.

```python
mae(pred, true) -> per-coordinate MAE
rmse(pred, true) -> per-coordinate RMSE
bias(pred, true) -> per-coordinate mean signed error (pred - true)
bias_corrected_mae(pred, true) -> MAE after removing each coordinate's mean offset
                                  within a trial
wrapped_difference(pred, true) -> shortest signed angular difference, in degrees
subject_bootstrap_ci(values_by_subject, n_boot=10000, ci=0.95, seed=0)
    -> (point_estimate, low, high); resample whole subjects, never individual frames
```

Requirements:
- Raw and bias-corrected results are always distinct and clearly named.
- Include tests with hand-computed expected values, written out in the test file, covering NaN handling, angle wraparound (179° vs -179° is 2° apart), an all-NaN coordinate, mismatched shapes (must raise an error) and bootstrap reproducibility from the seed.
- Show the actual `pytest` output. If you can't run the tests, say so plainly.

## Task 4: OpenCap dataset loader
Only start this if Task 1 confirms public access and you can see the file layout. Don't guess the layout.

Create `benchmarks/loaders/opencap_dataset.py`, which reads `.mot` joint angles and marker-based reference data for one subject and trial and returns arrays compatible with Task 3. Add tests using small synthetic fixtures. Nothing downloaded goes in the repo.

## When you finish, or when time runs out
Give a short report:
- What's complete and what isn't.
- Anything you couldn't verify.
- The three findings most likely to change our plan.

Unfinished work is fine. A clearly labelled gap is far more useful than a confident guess.
