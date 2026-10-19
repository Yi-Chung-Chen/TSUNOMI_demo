# M8b: location-conditioned state density

| score | group | token_count | mean | median | p95 | auroc_drone_positive | airborne_10_04_p95 | drone_fraction_above_airborne_10_04_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transition_nll | drone | 146 | 3.3924 | 2.6578 | 7.8579 | — | 7.0834 | 0.0959 |
| transition_nll | aircraft-airborne (2026-10-04) | 9200 | 4.0679 | 3.8981 | 7.0834 | 0.3362 | 7.0834 | 0.0959 |
| transition_nll | aircraft-ground (2026-10-04) | 2638 | 2.0781 | 1.4581 | 5.7732 | 0.7597 | 7.0834 | 0.0959 |
| transition_nll | aircraft-airborne (2026-10-03) | 11097 | 4.4526 | 4.3127 | 7.5434 | 0.2829 | 7.0834 | 0.0959 |
| transition_nll | aircraft-ground (2026-10-03) | 5332 | 2.1235 | 1.5274 | 5.7660 | 0.7467 | 7.0834 | 0.0959 |
| location_height_nll | drone | 146 | 10.4435 | 9.2597 | 16.7045 | — | 8.4107 | 0.7534 |
| location_height_nll | aircraft-airborne (2026-10-04) | 9200 | 6.7972 | 6.5437 | 8.4107 | 0.9660 | 8.4107 | 0.7534 |
| location_height_nll | aircraft-ground (2026-10-04) | 2638 | 5.5276 | 5.4438 | 7.4257 | 0.9886 | 8.4107 | 0.7534 |
| location_height_nll | aircraft-airborne (2026-10-03) | 11097 | 7.5284 | 7.4893 | 9.3827 | 0.8814 | 8.4107 | 0.7534 |
| location_height_nll | aircraft-ground (2026-10-03) | 5332 | 5.5188 | 4.6127 | 8.2238 | 0.9671 | 8.4107 | 0.7534 |
| speed_nll | drone | 146 | 6.2654 | 7.1050 | 8.9727 | — | 2.2724 | 1.0000 |
| speed_nll | aircraft-airborne (2026-10-04) | 9200 | 1.3131 | 1.1041 | 2.2724 | 0.9958 | 2.2724 | 1.0000 |
| speed_nll | aircraft-ground (2026-10-04) | 2638 | 1.4777 | 1.3264 | 2.8589 | 0.9942 | 2.2724 | 1.0000 |
| speed_nll | aircraft-airborne (2026-10-03) | 11097 | 1.4084 | 1.1651 | 2.7797 | 0.9928 | 2.2724 | 1.0000 |
| speed_nll | aircraft-ground (2026-10-03) | 5332 | 1.4663 | 1.2108 | 3.1672 | 0.9858 | 2.2724 | 1.0000 |
| course_nll | drone | 146 | 5.4686 | 6.1092 | 8.9724 | — | 1.7733 | 0.8493 |
| course_nll | aircraft-airborne (2026-10-04) | 9200 | 1.1391 | 1.0158 | 1.7733 | 0.9801 | 1.7733 | 0.8493 |
| course_nll | aircraft-ground (2026-10-04) | 2638 | 1.8801 | 1.6944 | 3.0590 | 0.8723 | 1.7733 | 0.8493 |
| course_nll | aircraft-airborne (2026-10-03) | 11097 | 1.2361 | 1.0844 | 2.0605 | 0.9663 | 1.7733 | 0.8493 |
| course_nll | aircraft-ground (2026-10-03) | 5332 | 1.9569 | 1.6217 | 2.8935 | 0.8689 | 1.7733 | 0.8493 |
| state_nll | drone | 146 | 22.1775 | 22.8430 | 25.6786 | — | 11.3880 | 1.0000 |
| state_nll | aircraft-airborne (2026-10-04) | 9200 | 9.2493 | 8.8326 | 11.3880 | 0.9998 | 11.3880 | 1.0000 |
| state_nll | aircraft-ground (2026-10-04) | 2638 | 8.8854 | 8.6435 | 11.1220 | 0.9999 | 11.3880 | 1.0000 |
| state_nll | aircraft-airborne (2026-10-03) | 11097 | 10.1729 | 9.8929 | 12.4621 | 0.9992 | 11.3880 | 1.0000 |
| state_nll | aircraft-ground (2026-10-03) | 5332 | 8.9420 | 8.5940 | 12.7418 | 0.9991 | 11.3880 | 1.0000 |
| combined_score | drone | 146 | 6.9572 | 7.3268 | 8.9018 | — | 1.3842 | 1.0000 |
| combined_score | aircraft-airborne (2026-10-04) | 9200 | 0.2904 | 0.1937 | 1.3842 | 0.9998 | 1.3842 | 1.0000 |
| combined_score | aircraft-ground (2026-10-04) | 2638 | -0.1675 | -0.2572 | 1.1458 | 0.9999 | 1.3842 | 1.0000 |
| combined_score | aircraft-airborne (2026-10-03) | 11097 | 0.6231 | 0.5229 | 1.7286 | 0.9992 | 1.3842 | 1.0000 |
| combined_score | aircraft-ground (2026-10-03) | 5332 | -0.0623 | -0.2583 | 1.9305 | 0.9991 | 1.3842 | 1.0000 |

## Plain-language conclusion

State NLL AUROC is 0.99976, compared with transition NLL 0.33624; combined AUROC is 0.99975. 100.0% of drone tokens exceed the10-04 airborne state-NLL p95, versus 9.6% for transition NLL and 100.0% for combined.

Location-conditioned state density flags this route much more consistently than transition NLL, which misses most of its steady flight.

Compare state and combined AUROC above; combining scores is not guaranteed to improve ranking.

The largest mean component excess over airborne aircraft is speed_nll: 4.952 nats. Component mean differences (drone minus10-04 airborne): location_height_nll +3.646, speed_nll +4.952, course_nll +4.330.

**This is one drone and one route: a demonstration, not a validated detector.** Neighbouring tokens are correlated and source interpolation differs between simulated drone poses and ADS-B aircraft. 10-04 p95 is a descriptive evaluation reference, not an independently validated deployment threshold.

## Fixed method and provenance

- Histograms fit only 1,007,315 eligible3Hz states from 618 train scenarios (09-28..10-02). Each state uses valid endpoints t−6 and t; intervening frames are not required by the plan. No validation/drone counts enter the density.
- Processed aircraft z and the existing interpolated drone z are already height above166m; no second altitude conversion. Speed/course use past2s displacement only. atan2(0,0)=0 for stationary course.
- Metadata geographic bounds use the M7 fixed-origin tangent-plane conversion.250m grid anchored at west/south ENU bounds; final cells are partial. Exact upper bounds are included in the final cells; all other out-of-bounds states use one separate outside bucket.
- Pool raw joint and conditional counts over3×3 spatial neighbours at fixed height with zero padding; outside pools only itself. Add alpha0.5 to every bin. Joint normalises over all(cell,height) including outside; speed/course normalise conditionally.
- User-authorized adjacent-bin smoothing: fixed[1,2,1]/4 on pooled histograms before alpha, circular for course and reflecting for speed. Raw/joint/height counts are unchanged. This is by bin index, not an equal physical m/s bandwidth; no125m/s tail representative is introduced. Hard bin lookup remains, so residual boundary steps can still occur.
- State NLL = location_height_nll + speed_nll + course_nll. Height boundaries15/100/300/600m; speed boundaries5/10/20/…/120m/s (14bins);12course bins begin at−pi.
- Standardisation fits only 16,42910-03 aircraft tokens, airborne+ground. Each score uses its own median/IQR; combined=max(z_transition,z_state). Parameters are in standardization.json. A zero IQR raises an error; no fallback/tuning.
- Scores join the exact M8 token rows at token_start_frame. State is causal at the start; transition uses the following2s GT and is available at token end. Combined is not an online score at token start.
- AUROC treats drone as positive and higher scores as anomalous, with average ranks for ties. Exceedance is strict > the separately computed10-04-airborne p95 of each metric.
- Optional combined-score GIFs use that same10-04 airborne combined p95 for the red outline; the labels show combined z-units, not NLL nats. State is known at token start, but the combined label appears only once its transition token completes.
- Original bins/spatial smoothing/combination follow the plan. The adjacent-bin correction was requested by Yi-Chung and fixed provisionally by the advisor before recomputing results; see QUESTIONS.md. No NN training/model changes, package installs or drone tuning.

Detailed counts/probabilities: state_density.npz; all token joins: token_scores.csv; run checks: acceptance.json.

## Smoothing correction: before / after

Fixed adjacent-bin[1,2,1]/4 kernel: circular course, reflecting speed, then alpha. No physical-distance bandwidth, new height/joint smoothing or drone-dependent parameter. Original outputs remain untouched. Hard bin lookup still permits residual steps.

aa0300 token28–30s: combined 1.6571 → 1.3848; old cutoff 1.4667, recalibrated cutoff 1.3842; flagged before/after = True/True. Course NLL 2.7272 → 2.7333; speed NLL 2.3578 → 2.0498.

| Group | Before combined flags | After, frozen old calibration/cutoff | After, recalibrated |
|---|---:|---:|---:|
| drone | 146/146 (100.00%) | 146 (100.00%) | 146 (100.00%) |
| aircraft-airborne (2026-10-04) | 460/9200 (5.00%) | 436 (4.74%) | 459 (4.99%) |
| aircraft-ground (2026-10-04) | 75/2638 (2.84%) | 76 (2.88%) | 81 (3.07%) |
| aircraft-airborne (2026-10-03) | 1249/11097 (11.26%) | 1064 (9.59%) | 1106 (9.97%) |
| aircraft-ground (2026-10-03) | 469/5332 (8.80%) | 374 (7.01%) | 379 (7.11%) |

Recalibrated10-04 airborne exceedance is approximately5% by construction (strict ties can make it slightly lower); that is not evidence of an improved false-positive rate. Frozen combined comparisons use both the old10-03 medians/IQRs and old cutoff, rather than mixing calibration scales. Normal-aircraft flags are false alerts under the clean-aircraft assumption. The corrected example is not a claim that all false alerts disappear. Full per-component comparisons and case timeline are in smoothing_comparison.csv and aa0300_comparison.csv.
