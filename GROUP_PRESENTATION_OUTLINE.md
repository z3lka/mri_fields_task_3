# MRIxFields2026 — Group Progress Presentation

Concise slide content. Text in *italics* is a visual suggestion or presenter note, not slide body text.

## Slide 1 — MRIxFields2026: Current Status

**Cross-Field MRI Translation and Harmonization**

- Group scope: Task 1, Task 2, and Task 3
- Goal: translate MRI across 0.1T, 1.5T, 3T, 5T, and 7T
- Modalities: T1W, T2W, and T2FLAIR
- Task 1/2 status: *add the responsible team members' latest verified status*
- Task 3 status: official baseline completed; 180 predictions validated
- Task 3 archive is ready for manual upload; no leaderboard score yet

*Visual: five field-strength labels connected by bidirectional arrows, with a small “Task 3 baseline ready” status badge.*

## Slide 2 — Why Cross-Field MRI Translation?

- MRI appearance changes with magnetic field strength
- Higher fields generally provide more signal that can support finer resolution or shorter scans
- Noise, tissue contrast, artifacts, and relaxation behavior also change across fields
- Translation aims to reduce scanner/field differences while preserving the same anatomy
- Potential benefit: more consistent images for clinical review and downstream AI

*Visual: show the same subject and modality at low, clinical, and ultra-high field (for example 0.1T → 3T → 7T). Add only three callouts: SNR, contrast, fine detail. Use an organizer-provided paired example or our paired training data; do not use unrelated patients.*

## Slide 3 — MRIxFields2026 Challenge Overview

- Five field strengths: 0.1T, 1.5T, 3T, 5T, 7T
- Three modalities: T1W, T2W, T2FLAIR
- Retrospective cohort: unpaired scans from different subjects
- Prospective cohort: travelling volunteers scanned across all fields
- Overall objective: accurate, realistic, and anatomically faithful cross-field synthesis

*Visual: a 3 × 5 grid: modalities as rows and field strengths as columns. Add paired/unpaired icons beneath it.*

## Slide 4 — Three Tasks, One Challenge

| Task | Translation | Main idea |
|---|---|---|
| Task 1 | 0.1T/1.5T/3T/5T → 7T | Synthesize a common ultra-high-field target |
| Task 2 | 0.1T → 1.5T/3T/5T/7T | Enhance ultra-low-field MRI |
| **Task 3 — our scope** | **Any field → any other field** | **One controllable unified model** |

*Visual: Task 1 as several arrows into 7T; Task 2 as arrows out of 0.1T; Task 3 as a five-node fully connected diagram. Highlight only Task 3.*

## Slide 5 — Dataset and Preprocessing

- Data organization: split → modality → field strength → NIfTI volume
- Full normalized volume: `364 × 436 × 364`, float32, values in `[0,1]`
- Common 0.5 mm isotropic MNI space with spatial metadata preserved
- Prospective data supports paired comparison; retrospective data is unpaired
- Baseline processing: axial 2D slices, pad/crop to `512 × 512`, then reconstruct the volume

```text
validation_prospective/
├── T1W/{0.1T, 1.5T, 3T, 5T, 7T}/
├── T2W/{0.1T, 1.5T, 3T, 5T, 7T}/
└── T2FLAIR/{0.1T, 1.5T, 3T, 5T, 7T}/
```

*Presenter note: public validation targets are private, so different validation IDs must not be treated as paired images.*

## Slide 6 — Evaluation and Postprocessing

- **nRMSE ↓:** normalized voxel-level reconstruction error
- **SSIM ↑:** structural similarity and local image consistency
- **LPIPS ↓:** learned perceptual difference
- **Task 1/2:** also use Dice and normalized volume consistency for anatomical preservation
- **SynthSeg (Task 1/2):** segments brain structures from generated MRI
- Segmentation lets us check whether anatomy and regional volumes remain consistent
- **Task 3:** image-quality metrics only; no SynthSeg or segmentation output

*Visual: left side = nRMSE/SSIM/LPIPS; right side = generated MRI → SynthSeg → anatomical labels → Dice/volume. Add “Task 1/2 only” above the right side.*

## Slide 7 — Baseline Models

- **CUT:** one-way unpaired translation using contrastive feature learning
- **CycleGAN:** unpaired translation using forward/backward cycle consistency
- **U-Net:** encoder–decoder commonly used for paired image-to-image reconstruction
- **StarGAN v2:** multi-domain translation controlled by the requested target domain
- **Our Task 3 baseline:** one shared StarGAN v2 checkpoint for every modality and field direction
- **Initial internal models:** *add Furkan's model names and one-line descriptions after verification*

*Visual: four small model cards. Highlight which models belong to each task and avoid detailed architecture diagrams.*

## Slide 8 — Experiment and Submission Pipeline

```text
Validation MRI
      ↓
Input audit + case planning
      ↓
Shared StarGAN v2 checkpoint
      ↓
Full-volume translated MRI
      ↓
Numerical + spatial validation
      ↓
SynthSeg for Task 1/2 only
      ↓
Official crop and folder builder
      ↓
task1.zip / task2.zip / task3.zip
```

- Task 1: four source fields → 7T, across three modalities
- Task 2: 0.1T → four higher fields, across three modalities
- Task 3: all 20 directed non-identity pairs using one unified model
- Predictions are checked, cropped, renamed, and packaged in the official format

## Slide 9 — Validation Results

| Task | nRMSE ↓ | SSIM ↑ | LPIPS ↓ | Dice ↑ | Volume consistency ↑ |
|---|---:|---:|---:|---:|---:|
| Task 1 | TBD | TBD | TBD | TBD | TBD |
| Task 2 | TBD | TBD | TBD | TBD | TBD |
| Task 3 | Awaiting upload | Awaiting upload | Awaiting upload | N/A | N/A |

**Verified Task 3 pipeline status:** 180/180 predictions, 60/60 pair folders, valid float32 range, spatial metadata, crop shape, and archive structure.

*Presenter note: replace “TBD” only with verified local holdout or leaderboard results. Do not mix structural pipeline checks with image-quality scores.*

## Slide 10 — Qualitative Observations

*Use a three-column panel for one or two representative cases:*

| Source input | Generated target | Ground truth / SynthSeg overlay |
|---|---|---|
| Field and modality label | Same subject, requested field | Use paired data; SynthSeg only for Task 1/2 |

Suggested observations:

- Is anatomy preserved across translation?
- Does contrast move in the expected target-field direction?
- Are fine structures retained or oversmoothed?
- Are there hallucinated structures, edge artifacts, or intensity clipping?
- Do observations agree across T1W, T2W, and T2FLAIR?
- For Task 1/2, are SynthSeg regions and boundaries anatomically plausible?

*Do not claim qualitative success until these panels have been inspected. Public validation inputs do not include paired targets.*

## Slide 11 — Conclusion and Next Steps

- Established workflows for the three complementary translation settings
- Task 1 focuses on a common 7T target; Task 2 on 0.1T enhancement
- Task 3 demonstrates controllable any-to-any translation with one shared model
- Verified Task 3 baseline and submission archive; Task 1/2 status must be added by their owners
- Current limitation: incomplete cross-task quantitative and qualitative comparison
- Next: collect verified results, upload valid submissions, and compare failure modes

## Presenter reference notes

- Challenge overview and dataset: <https://mrixfields.chihucloud.com/2026/>
- Current task definitions and Task 3 constraints: <https://mrixfields.chihucloud.com/2026/tasks.html>
- Official submission rules: <https://github.com/MRIxFields/MRIxFields2026/blob/main/Submission/README.md>
- MRI field-strength background: Marques et al., *Low-field MRI: An MR physics perspective*: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6590434/>
- Local Task 3 status evidence: `submission_ready/task3_validation_report.json`
