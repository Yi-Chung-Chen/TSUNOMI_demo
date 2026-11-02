# M10 runway-anchor attention experiment

Both alignment gates passed: reference stdout byte-identical; exact-ENU counts within±5%. Both fresh runs use seed817,20epochs,batch32/16,workers4 and the original optimizer/lr/dropout. Checkpoint selection is best10-03smoothed val loss only. The old epoch016 came from a128epoch schedule. No hyperparameter search, identity dropout or demo-checkpoint replacement. All final evaluations run on CPU.

## Training selection

| Run | Best epoch | Best 10-03 val loss |
|---|---:|---:|
| base20 | 17 | 4.315308 |
| anchor | 18 | 4.249187 |

## Primary comparisons (10-04)

Point estimates only; better/worse is a numerical comparison, not statistical significance. Lower CE/trajectory errors/miss rates and higher accuracy/AUROC are better. M8 means/medians are descriptive likelihoods and not labelled better/worse.

| Family | Group | Horizon | Axis | Metric | epoch016 | base20 | anchor | CV | Anchor vs base20 | Base20 vs epoch016 |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| CE | all |  |  | loss | 4.14411 | 4.11712 | 4.03180 | — | better | better |
| CE | all |  |  | ce | 3.49720 | 3.46414 | 3.36371 | — | better | better |
| CE | all |  |  | accuracy | 0.18751 | 0.19227 | 0.20462 | — | better | better |
| M6 | all | 10 | horizontal | minADE_m | 8.84079 | 8.04665 | 6.86438 | 26.89490 | better | better |
| M6 | all | 10 | horizontal | ADE_m | 16.85722 | 16.11546 | 14.42503 | 26.89490 | better | better |
| M6 | all | 10 | horizontal | FDE_m | 36.28070 | 34.16462 | 29.15189 | 60.55865 | better | better |
| M6 | all | 10 | horizontal | miss_rate | — | — | — | — | NA | NA |
| M6 | all | 10 | horizontal | mean_rollout_miss_rate | — | — | — | — | NA | NA |
| M6 | all | 10 | vertical | minADE_m | 2.95769 | 2.89498 | 2.73230 | 14.23825 | better | better |
| M6 | all | 10 | vertical | ADE_m | 5.48269 | 5.27279 | 5.16685 | 14.23825 | better | better |
| M6 | all | 10 | vertical | FDE_m | 9.57660 | 9.08490 | 8.69310 | 28.35340 | better | better |
| M6 | all | 10 | vertical | miss_rate | — | — | — | — | NA | NA |
| M6 | all | 10 | vertical | mean_rollout_miss_rate | — | — | — | — | NA | NA |
| M6 | all | 20 | horizontal | minADE_m | 20.85026 | 18.45178 | 14.60297 | 70.18083 | better | better |
| M6 | all | 20 | horizontal | ADE_m | 43.50814 | 40.11735 | 32.62598 | 70.18083 | better | better |
| M6 | all | 20 | horizontal | FDE_m | 107.84949 | 98.22482 | 76.26222 | 173.80270 | better | better |
| M6 | all | 20 | horizontal | miss_rate | — | — | — | — | NA | NA |
| M6 | all | 20 | horizontal | mean_rollout_miss_rate | — | — | — | — | NA | NA |
| M6 | all | 20 | vertical | minADE_m | 5.10924 | 4.89810 | 4.29719 | 29.98585 | better | better |
| M6 | all | 20 | vertical | ADE_m | 9.78934 | 9.25268 | 8.32383 | 29.98585 | better | better |
| M6 | all | 20 | vertical | FDE_m | 19.06253 | 17.90073 | 14.67347 | 63.24374 | better | better |
| M6 | all | 20 | vertical | miss_rate | — | — | — | — | NA | NA |
| M6 | all | 20 | vertical | mean_rollout_miss_rate | — | — | — | — | NA | NA |
| M6 | all | 30 | horizontal | minADE_m | 39.48603 | 34.36625 | 25.31010 | 131.15234 | better | better |
| M6 | all | 30 | horizontal | ADE_m | 84.00595 | 77.46078 | 58.05507 | 131.15234 | better | better |
| M6 | all | 30 | horizontal | FDE_m | 225.98863 | 208.27410 | 148.04810 | 345.05390 | better | better |
| M6 | all | 30 | horizontal | miss_rate | 0.04502 | 0.03318 | 0.02844 | 0.21801 | better | better |
| M6 | all | 30 | horizontal | mean_rollout_miss_rate | 0.13300 | 0.11271 | 0.06391 | 0.21801 | better | better |
| M6 | all | 30 | vertical | minADE_m | 7.25836 | 6.93800 | 5.25731 | 49.30798 | better | better |
| M6 | all | 30 | vertical | ADE_m | 14.02340 | 13.08290 | 10.41892 | 49.30798 | better | better |
| M6 | all | 30 | vertical | FDE_m | 28.89003 | 26.66446 | 18.67389 | 107.27791 | better | better |
| M6 | all | 30 | vertical | miss_rate | — | — | — | — | NA | NA |
| M6 | all | 30 | vertical | mean_rollout_miss_rate | — | — | — | — | NA | NA |
| M8 | aircraft-airborne (2026-10-04) |  |  | nll_auroc | 0.33624 | 0.45347 | 0.54998 | — | better | better |
| M8 | aircraft-ground (2026-10-04) |  |  | nll_auroc | 0.75967 | 0.83414 | 0.88359 | — | better | better |
| Flow | east/airborne | 30 | horizontal | FDE_m | 403.85330 | 372.55605 | 292.31363 | 463.75775 | better | better |
| Flow | east/airborne | 30 | vertical | FDE_m | 43.60673 | 38.88406 | 23.34085 | 124.16830 | better | better |
| Flow | east/all | 30 | horizontal | FDE_m | 370.02391 | 352.63674 | 245.89219 | 436.35859 | better | better |
| Flow | east/all | 30 | vertical | FDE_m | 37.88928 | 34.97533 | 23.22150 | 80.03251 | better | better |
| Flow | east/ground | 30 | horizontal | FDE_m | 335.34877 | 332.21946 | 198.31022 | 408.27445 | better | better |
| Flow | east/ground | 30 | vertical | FDE_m | 32.02890 | 30.96889 | 23.09917 | 34.79333 | better | better |
| Flow | west/airborne | 30 | horizontal | FDE_m | 137.46976 | 126.37267 | 94.79193 | 269.81307 | better | better |
| Flow | west/airborne | 30 | vertical | FDE_m | 21.96663 | 19.82064 | 16.16410 | 125.99058 | better | better |
| Flow | west/all | 30 | horizontal | FDE_m | 177.05532 | 158.38762 | 110.18169 | 296.86190 | better | better |
| Flow | west/all | 30 | vertical | FDE_m | 22.69674 | 20.96117 | 15.83505 | 109.74534 | better | better |
| Flow | west/ground | 30 | horizontal | FDE_m | 340.34574 | 290.44925 | 173.66445 | 408.43833 | better | better |
| Flow | west/ground | 30 | vertical | FDE_m | 25.70845 | 25.66587 | 14.47774 | 42.73373 | better | better |

## Flow definition and limitations

Provisional advisor clarification: exact supplied flow-reference script, not centreline masks. For each track,6-valid-frame (nominal2s) displacements with on_ground at both endpoints and speed>30m/s; require>=3roll frames, then median normalised dx. abs(dx)>0.9 votes east/west; scenario majority wins, ties/no-roll skipped. The direction filter stands in for six E-W runways; diagonal runway rolls are excluded by direction, not a centreline mask.

10-03 flow scenarios: {'east': 68, 'none/tie': 23}. 10-04: {'west': 39, 'none/tie': 37, 'east': 15}. Selection is entirely east-flow on10-03 and may favour east-flow behaviour; the same selection rule applies to both new runs. 10-04 has been inspected during prior demo development and is not an untouched test set.

All/all-airborne/ground and both axes at10/20/30s, plus every comparison and delta, are in summary.csv. M6 errors use unchanged32-rollout topk5/temp1 metrics; minADE is per-agent whole-path minimum; miss_rate requires ALL32 horizontal30s endpoints>500m; mean_rollout_miss_rate is also reported. Counts are token or eligible agent-window counts, not independent scenarios.

M8 reuses the exact28413original token rows, unchanged GT token chain and residuals. AUROC treats the single native-pose drone route as positive, higher NLL as anomalous. This is one route and smoothness/acquisition differ from aircraft; no general drone detector or calibrated threshold is established.

Attention figure uses observed6s history only and a fixed seed of817 for three10-04windows. Weights average heads and all4anchor layers; they are diagnostic associations, not causal explanations. Basemap tiles are existing local cache only; no downloads.

The integrated demo continues using frozen epoch_016.ckpt.

## Plain-language result

Runway anchors improve the primary30s horizontal mean FDE from 208.27m to 148.05m (28.9% lower) relative to the matched20epoch baseline. The schedule-only baseline also improves from the old checkpoint's 225.99m, so these two effects are reported separately.

Drone NLL AUROC versus10-04 airborne aircraft improves from 0.453 to 0.550 (old epoch016:0.336); versus ground aircraft it improves from 0.834 to 0.884. The airborne AUROC remains only slightly above chance, so this is not an effective standalone drone detector. Forecasting gains are stronger evidence than the anomaly result, and the experiment still covers only one drone route.

Not every metric improves. The following10-04 subgroup results are worse than base20:

- M6 / airborne / 30s / horizontal / miss_rate: 0.034591 -> 0.037736 (n=318).
