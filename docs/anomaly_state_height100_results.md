# M8b: location-conditioned state density

## Height100 revision result

Merging the two lowest raw height layers removes the selected blue-aircraft
alert run, but does **not** improve every normal-aircraft metric. The11 GIFs
are a separate comparison variant; both preceding versions are preserved.

- aabc11/scenario000667, token starts238–274s (10-03): flags8/19 →0/19,
  longest run16s →0s. aa0300/scenario000536 starts24–60s (10-03):5/19 →0/19.
- Frozen baseline10-03 scaling and cutoff:10-04 all aircraft540/11838 →467/11838;
  airborne459/9200 →388/9200; ground81/2638 →79/2638.
- Recalibrated10-03 scaling and10-04 airborne p95: all aircraft572/11838;
  ground112/2638 (4.25%, versus3.07% baseline). Ground max run18s →60s;
  tracks with a ground run≥6s6/149 →7/149. Median ground run remains2s.
  Airborne flags460/9200 are5% by construction, not independent improvement.
- Drone combined flags146/146 →138/146 (94.52%); all8 misses are below100m.
  Below100m median state NLL18.3992 →13.5477 (32 tokens,24flagged);
  100–300m23.04183 →23.04176 (114tokens,114flagged).
- Combined AUROC versus10-04 airborne falls0.999751 →0.992693.

The broader low-height bucket admits more slow traffic, explaining both the
selected taxi improvement and weaker low-altitude drone separation. With the
usual recalibration, ground false alerts and their longest run increase. This
is a tradeoff requiring review, not a demonstrated overall detector upgrade.
No extra threshold tuning or compensating score was added.

| score | group | token_count | mean | median | p95 | auroc_drone_positive | airborne_10_04_p95 | drone_fraction_above_airborne_10_04_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transition_nll | drone | 146 | 3.3924 | 2.6578 | 7.8579 | — | 7.0834 | 0.0959 |
| transition_nll | aircraft-airborne (2026-10-04) | 9200 | 4.0679 | 3.8981 | 7.0834 | 0.3362 | 7.0834 | 0.0959 |
| transition_nll | aircraft-ground (2026-10-04) | 2638 | 2.0781 | 1.4581 | 5.7732 | 0.7597 | 7.0834 | 0.0959 |
| transition_nll | aircraft-airborne (2026-10-03) | 11097 | 4.4526 | 4.3127 | 7.5434 | 0.2829 | 7.0834 | 0.0959 |
| transition_nll | aircraft-ground (2026-10-03) | 5332 | 2.1235 | 1.5274 | 5.7660 | 0.7467 | 7.0834 | 0.0959 |
| location_height_nll | drone | 146 | 10.0865 | 8.8016 | 16.7045 | — | 8.2388 | 0.6027 |
| location_height_nll | aircraft-airborne (2026-10-04) | 9200 | 6.5718 | 6.4415 | 8.2388 | 0.9089 | 8.2388 | 0.6027 |
| location_height_nll | aircraft-ground (2026-10-04) | 2638 | 5.3689 | 5.2088 | 7.0942 | 0.9740 | 8.2388 | 0.6027 |
| location_height_nll | aircraft-airborne (2026-10-03) | 11097 | 7.2559 | 7.1325 | 9.3591 | 0.8210 | 8.2388 | 0.6027 |
| location_height_nll | aircraft-ground (2026-10-03) | 5332 | 5.3554 | 4.5315 | 7.8464 | 0.9557 | 8.2388 | 0.6027 |
| speed_nll | drone | 146 | 5.8994 | 6.2154 | 8.9727 | — | 2.5172 | 0.9452 |
| speed_nll | aircraft-airborne (2026-10-04) | 9200 | 1.3890 | 1.1349 | 2.5172 | 0.9868 | 2.5172 | 0.9452 |
| speed_nll | aircraft-ground (2026-10-04) | 2638 | 1.5897 | 1.3867 | 3.1092 | 0.9745 | 2.5172 | 0.9452 |
| speed_nll | aircraft-airborne (2026-10-03) | 11097 | 1.5073 | 1.2385 | 2.8536 | 0.9790 | 2.5172 | 0.9452 |
| speed_nll | aircraft-ground (2026-10-03) | 5332 | 1.5783 | 1.3110 | 3.1858 | 0.9641 | 2.5172 | 0.9452 |
| course_nll | drone | 146 | 5.2372 | 4.8675 | 8.9724 | — | 1.8560 | 0.8904 |
| course_nll | aircraft-airborne (2026-10-04) | 9200 | 1.1849 | 1.0279 | 1.8560 | 0.9763 | 1.8560 | 0.8904 |
| course_nll | aircraft-ground (2026-10-04) | 2638 | 1.9034 | 1.6995 | 3.1488 | 0.8739 | 1.8560 | 0.8904 |
| course_nll | aircraft-airborne (2026-10-03) | 11097 | 1.2504 | 1.1123 | 1.9131 | 0.9658 | 1.8560 | 0.8904 |
| course_nll | aircraft-ground (2026-10-03) | 5332 | 2.0125 | 1.6813 | 3.2025 | 0.8593 | 1.8560 | 0.8904 |
| state_nll | drone | 146 | 21.2230 | 21.8284 | 25.6785 | — | 10.8370 | 0.9795 |
| state_nll | aircraft-airborne (2026-10-04) | 9200 | 9.1457 | 8.8039 | 10.8370 | 0.9960 | 10.8370 | 0.9795 |
| state_nll | aircraft-ground (2026-10-04) | 2638 | 8.8620 | 8.6077 | 11.0280 | 0.9956 | 10.8370 | 0.9795 |
| state_nll | aircraft-airborne (2026-10-03) | 11097 | 10.0136 | 9.8062 | 12.1826 | 0.9879 | 10.8370 | 0.9795 |
| state_nll | aircraft-ground (2026-10-03) | 5332 | 8.9462 | 8.5632 | 13.0281 | 0.9902 | 10.8370 | 0.9795 |
| combined_score | drone | 146 | 7.3887 | 7.7599 | 10.2039 | — | 1.3552 | 0.9452 |
| combined_score | aircraft-airborne (2026-10-04) | 9200 | 0.2652 | 0.1571 | 1.3552 | 0.9927 | 1.3552 | 0.9452 |
| combined_score | aircraft-ground (2026-10-04) | 2638 | -0.1630 | -0.2962 | 1.1954 | 0.9952 | 1.3552 | 0.9452 |
| combined_score | aircraft-airborne (2026-10-03) | 11097 | 0.6133 | 0.4783 | 1.7915 | 0.9848 | 1.3552 | 0.9452 |
| combined_score | aircraft-ground (2026-10-03) | 5332 | -0.0290 | -0.3078 | 2.1736 | 0.9899 | 1.3552 | 0.9452 |

## Plain-language conclusion

State NLL AUROC is 0.99603, compared with transition NLL 0.33624; combined AUROC is 0.99269. 97.9% of drone tokens exceed the10-04 airborne state-NLL p95, versus 9.6% for transition NLL and 94.5% for combined.

Location-conditioned state density flags this route much more consistently than transition NLL, which misses most of its steady flight.

Compare state and combined AUROC above; combining scores is not guaranteed to improve ranking.

The largest mean component excess over airborne aircraft is speed_nll: 4.510 nats. Component mean differences (drone minus10-04 airborne): location_height_nll +3.515, speed_nll +4.510, course_nll +4.052.

**This is one drone and one route: a demonstration, not a validated detector.** Neighbouring tokens are correlated and source interpolation differs between simulated drone poses and ADS-B aircraft. 10-04 p95 is a descriptive evaluation reference, not an independently validated deployment threshold.

## Fixed method and provenance

- Histograms fit only 1,007,315 eligible3Hz states from 618 train scenarios (09-28..10-02). Each state uses valid endpoints t−6 and t; intervening frames are not required by the plan. No validation/drone counts enter the density.
- Processed aircraft z and the existing interpolated drone z are already height above166m; no second altitude conversion. Speed/course use past2s displacement only. atan2(0,0)=0 for stationary course.
- Metadata geographic bounds use the M7 fixed-origin tangent-plane conversion.250m grid anchored at west/south ENU bounds; final cells are partial. Exact upper bounds are included in the final cells; all other out-of-bounds states use one separate outside bucket.
- Pool raw joint and conditional counts over3×3 spatial neighbours at fixed height with zero padding; outside pools only itself. Add alpha0.5 to every bin. Joint normalises over all(cell,height) including outside; speed/course normalise conditionally.
- User-authorized adjacent-bin smoothing: fixed[1,2,1]/4 on pooled histograms before alpha, circular for course and reflecting for speed. Raw/joint/height counts are unchanged. This is by bin index, not an equal physical m/s bandwidth; no125m/s tail representative is introduced. Hard bin lookup remains, so residual boundary steps can still occur.
- State NLL = location_height_nll + speed_nll + course_nll. Height boundaries100/300/600m; speed boundaries5/10/20/…/120m/s (14bins);12course bins begin at−pi.
- Standardisation fits only 16,42910-03 aircraft tokens, airborne+ground. Each score uses its own median/IQR; combined=max(z_transition,z_state). Parameters are in standardization.json. A zero IQR raises an error; no fallback/tuning.
- Scores join the exact M8 token rows at token_start_frame. State is causal at the start; transition uses the following2s GT and is available at token end. Combined is not an online score at token start.
- AUROC treats drone as positive and higher scores as anomalous, with average ranks for ties. Exceedance is strict > the separately computed10-04-airborne p95 of each metric.
- Optional combined-score GIFs use that same10-04 airborne combined p95 for the red outline; the labels show combined z-units, not NLL nats. State is known at token start, but the combined label appears only once its transition token completes.
- Original bins/spatial smoothing/combination follow the plan. The adjacent-bin correction was requested by Yi-Chung and fixed provisionally by the advisor before recomputing results; see QUESTIONS.md. No NN training/model changes, package installs or drone tuning.
- Height100 revision merges only the first two raw height bins before smoothing/pseudocounts.10-04 and the shown aircraft cases informed this revision, so10-04 is inspected development data and must not be presented as a pristine holdout. A future independent dataset is needed for validation.

Detailed counts/probabilities: state_density.npz; all token joins: token_scores.csv; run checks: acceptance.json.

## Height <100m merge: before / after

First two raw height bins merged before pooling and alpha. Stationary course remains atan2(0,0)=0; those counts now enter the <100m layer. Fixed adjacent-bin[1,2,1]/4 kernel: circular course, reflecting speed, then alpha. No physical-distance bandwidth or drone-dependent parameter. Original outputs remain untouched. Hard bin lookup still permits residual steps.

aa0300 token28–30s: combined 1.3848 → 0.8834; old cutoff 1.3842, recalibrated cutoff 1.3552; flagged before/after = True/False. Course NLL 2.7333 → 1.5843; speed NLL 2.0498 → 2.5701.

| Group | Before combined flags | After, frozen old calibration/cutoff | After, recalibrated |
|---|---:|---:|---:|
| drone | 146/146 (100.00%) | 138 (94.52%) | 138 (94.52%) |
| aircraft-airborne (2026-10-04) | 459/9200 (4.99%) | 388 (4.22%) | 460 (5.00%) |
| aircraft-ground (2026-10-04) | 81/2638 (3.07%) | 79 (2.99%) | 112 (4.25%) |
| aircraft-airborne (2026-10-03) | 1106/11097 (9.97%) | 956 (8.61%) | 1267 (11.42%) |
| aircraft-ground (2026-10-03) | 379/5332 (7.11%) | 402 (7.54%) | 441 (8.27%) |

Recalibrated10-04 airborne exceedance is approximately5% by construction (strict ties can make it slightly lower); that is not evidence of an improved false-positive rate. Frozen combined comparisons use both the old10-03 medians/IQRs and old cutoff, rather than mixing calibration scales. Normal-aircraft flags are false alerts under the clean-aircraft assumption. The corrected example is not a claim that all false alerts disappear. Full per-component comparisons and case timeline are in height_comparison.csv and aa0300_comparison.csv.

## Consecutive alerts

Runs group by (scenario, agent); missing/unflagged tokens break runs, and starts must differ by exactly 6 frames. Duration = 2s × token count. Ground/airborne subgroup runs break at phase changes; aircraft-all includes them. These are scenario-agent tracks, not independent flights.

| Group | Variant | Flags | Max run s | Median run s | Tracks with ≥6s run |
|---|---|---:|---:|---:|---:|
| drone | before | 146/146 | 292 | 292 | 1/1 (100.00%) |
| drone | after_frozen | 138/146 | 246 | 14 | 1/1 (100.00%) |
| drone | after_recalibrated | 138/146 | 246 | 14 | 1/1 (100.00%) |
| aircraft-airborne (2026-10-04) | before | 459/9200 | 48 | 2 | 29/354 (8.19%) |
| aircraft-airborne (2026-10-04) | after_frozen | 388/9200 | 48 | 2 | 18/354 (5.08%) |
| aircraft-airborne (2026-10-04) | after_recalibrated | 460/9200 | 48 | 2 | 28/354 (7.91%) |
| aircraft-ground (2026-10-04) | before | 81/2638 | 18 | 2 | 6/149 (4.03%) |
| aircraft-ground (2026-10-04) | after_frozen | 79/2638 | 18 | 2 | 7/149 (4.70%) |
| aircraft-ground (2026-10-04) | after_recalibrated | 112/2638 | 60 | 2 | 7/149 (4.70%) |
| aircraft-airborne (2026-10-03) | before | 1106/11097 | 60 | 2 | 85/410 (20.73%) |
| aircraft-airborne (2026-10-03) | after_frozen | 956/11097 | 60 | 2 | 61/410 (14.88%) |
| aircraft-airborne (2026-10-03) | after_recalibrated | 1267/11097 | 62 | 2 | 89/410 (21.71%) |
| aircraft-ground (2026-10-03) | before | 379/5332 | 152 | 2 | 17/260 (6.54%) |
| aircraft-ground (2026-10-03) | after_frozen | 402/5332 | 152 | 2 | 17/260 (6.54%) |
| aircraft-ground (2026-10-03) | after_recalibrated | 441/5332 | 152 | 2 | 20/260 (7.69%) |
| aircraft-all (2026-10-03) | before | 1485/16429 | 152 | 2 | 100/410 (24.39%) |
| aircraft-all (2026-10-03) | after_frozen | 1358/16429 | 152 | 2 | 77/410 (18.78%) |
| aircraft-all (2026-10-03) | after_recalibrated | 1708/16429 | 152 | 2 | 108/410 (26.34%) |
| aircraft-all (2026-10-04) | before | 540/11838 | 48 | 2 | 33/354 (9.32%) |
| aircraft-all (2026-10-04) | after_frozen | 467/11838 | 48 | 2 | 25/354 (7.06%) |
| aircraft-all (2026-10-04) | after_recalibrated | 572/11838 | 60 | 2 | 35/354 (9.89%) |

Drone height breakdown (same physical height groups before/after):

| Height m | Tokens | Before median state NLL | After median state NLL | After combined flags |
|---|---:|---:|---:|---:|
| <100 | 32 | 18.399162033022108 | 13.547735640055379 | 24 |
| 100–300 | 114 | 23.04183003160509 | 23.041763773640447 | 114 |
| 300–600 | 0 | None | None | 0 |
| >=600 | 0 | None | None | 0 |

The displayed aabc11 (scenario000667) and aa0300 (scenario000536) cases are dated10-03.10-04 aggregate results have also been inspected during revision, so they are development evidence. Full case timelines: aircraft_cases.csv; full group/run comparisons: alert_runs.csv and height_comparison.csv.
