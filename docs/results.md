# Task 3 results and evidence

This is the public, aggregate-only record behind the [README](../README.md).
The experiment wrap-up has an evidence cut-off of **24 September 2026**;
the source files and numbers below were rechecked locally on **12 October 2026**.
No training, inference, or external submission was rerun for this documentation.

## Recorded FPS-Former validation results

The project experiment CSV records three Task 3 FPS-Former submissions:

| Configured run | Recorded date | Submission ID | Overall nRMSE ↓ | Overall SSIM ↑ | Overall LPIPS ↓ |
| --- | --- | --- | ---: | ---: | ---: |
| Initial · 30 epochs | 28 August 2026 | `9778284` | 0.283937 | 0.864881 | 0.156116 |
| Target 150 | 31 August 2026 | `9778735` | 0.248285 | 0.878918 | 0.149620 |
| Target 200 | 2 September 2026 | `9779018` | 0.247053 | 0.879531 | 0.149198 |

“Best” means best overall among these three recorded FPS-Former submissions,
not best across the competition or across every model in the project log.
The values are transcribed from a locally preserved leaderboard-results CSV;
the live leaderboard was not independently authenticated during this update.
No final test score, competition placement, or award is established by these files.

The relative nRMSE reductions are 12.6% from initial to target 150, 0.5% from
target 150 to target 200, and 13.0% from initial to target 200. Relative reduction
is `(earlier - later) / earlier × 100`.

## Training labels and checkpoint selection

The configured training schedule progressed through target 100, target 150,
and target 200. The target-100 log used 128-pixel patches and batch size 1;
the later continuations used 256-pixel patches and batch size 16.
There is no separate target-100 leaderboard row in the supplied FPS-Former evidence.

These are not clean, independent runs of 150 or 200 epochs. Continuations
restored selected checkpoints, so their recorded epoch ranges overlap:

| Local training history | Recorded epoch range | Interpretation |
| --- | --- | --- |
| Target 100 | 14–100 | Earlier configuration; epochs 14–24 appear twice |
| Target 150 | 54–150 | Best local nRMSE at epoch 95 |
| Target 200 | 96–200 | Best local nRMSE at epoch 96 |

The wrap-up associates the target-150 archive with the epoch-95 checkpoint.
The evidence here does not independently establish the exact checkpoint epoch
used for submission `9779018`. The target labels must not be read as that claim.
CSV row labels also vary by modality within a submission; grouping by submission
ID and the documented schedule avoids treating those labels as separate runs.

## Three different kinds of evidence

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| Local paired-prospective holdout metrics | Slice-level diagnostic behavior during training | Organizer validation or final test quality |
| Structural and numerical archive checks | Correct files, counts, shape, dtype, range, and spatial metadata | Reconstruction quality or competition rank |
| Recorded public-validation scores | The three reported organizer-evaluated validation outcomes | Independently verified live scores or final test placement |

In the target-200 local branch, training L1 fell 38.8% from epoch 96 to 200,
while overall holdout nRMSE rose 14.2%; T2FLAIR nRMSE rose 56.9%.
Meanwhile the next public-validation submission improved slightly overall.
This suggests diminishing returns and a holdout/generalization mismatch;
it is not a controlled causal diagnosis of overfitting.

The target-200 training log records an L1 objective. LPIPS was measured for
evaluation; optional LPIPS-loss and augmentation recipes in the code are not
evidence that those features produced the reported scores.

## Historical submission checks

The wrap-up records completion of the official shared StarGAN v2 baseline
milestone and the FPS-Former submission pipeline. For the target-150 FPS-Former
archive, it specifically records a passing full-volume smoke case, 180
predictions, 60 modality/pair directories, and the required top-level `task3/`.
The recorded archive SHA-256 is:

```text
cbe53ee890b59cefb43145cb4d971f03d937d5510a64ee1a48c88edeb79f1e75
```

That archive was exported privately and is not present in this checkout.
This is historical evidence from the wrap-up, not a fresh artifact validation
or the verified hash of the target-200 submission. The prepared conditional
3D U-Net has no measured result in this implementation's wrap-up.

## Source provenance

Only Task 3 FPS-Former records were used for the score tables. Raw logs and the
full experiment CSV remain ignored; they are not required to browse this repo.
Their SHA-256 hashes matched the pinned wrap-up summary during the local check:

| Source | SHA-256 |
| --- | --- |
| Leaderboard-results CSV | `11775958528a0ce09ccb4f9ed2331bb51909f27536aa059d02db0f8c063dfad5` |
| Target-100 training log | `f05ad35a75223ba1f679399b22d5bbe3805239fe97a72f60345a6cbdb7977a0b` |
| Target-150 training log | `61c19a7cec565cfc2e7c796bb4d985b0161aaff1d87f3013b5df34ccfc2d26ec` |
| Target-200 training log | `2067aa089774884d0fcc4728696729b2d13cf4088765dc43579ee39862639a16` |

The local wrap-up records workspace commit
`d5be62ba2a140428f47b678ceeb8955d8968201e` at evidence collection. Official
baseline code/weight preparation pins are listed in the
[baseline guide](baseline.md); they are not a claim about the FPS-Former
submission checkpoint or the latest upstream revisions.

## Competition status note

This README reflects the user's completed competition participation. The live
[competition homepage](https://mrixfields.chihucloud.com/2026/) and
[tasks page](https://mrixfields.chihucloud.com/2026/tasks.html), checked on
12 October 2026, disagree on the scheduled final-results release date:
1 October and 8 October respectively. Neither date is used as proof of an
actual published final result. Until an organizer-issued result is available,
this repository reports the recorded validation scores without a final rank.
