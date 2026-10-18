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
| speed_nll | drone | 146 | 6.3610 | 7.2471 | 8.9727 | — | 3.1919 | 0.9110 |
| speed_nll | aircraft-airborne (2026-10-04) | 9200 | 1.2098 | 0.8789 | 3.1919 | 0.9874 | 3.1919 | 0.9110 |
| speed_nll | aircraft-ground (2026-10-04) | 2638 | 1.3928 | 1.2199 | 2.8834 | 0.9923 | 3.1919 | 0.9110 |
| speed_nll | aircraft-airborne (2026-10-03) | 11097 | 1.3438 | 1.0572 | 3.3271 | 0.9831 | 3.1919 | 0.9110 |
| speed_nll | aircraft-ground (2026-10-03) | 5332 | 1.4270 | 1.1321 | 3.3271 | 0.9820 | 3.1919 | 0.9110 |
| course_nll | drone | 146 | 6.2229 | 7.2457 | 8.9724 | — | 1.7126 | 0.9178 |
| course_nll | aircraft-airborne (2026-10-04) | 9200 | 0.8878 | 0.7617 | 1.7126 | 0.9758 | 1.7126 | 0.9178 |
| course_nll | aircraft-ground (2026-10-04) | 2638 | 1.6819 | 1.4715 | 3.6200 | 0.9179 | 1.7126 | 0.9178 |
| course_nll | aircraft-airborne (2026-10-03) | 11097 | 1.0582 | 0.9084 | 2.3248 | 0.9622 | 1.7126 | 0.9178 |
| course_nll | aircraft-ground (2026-10-03) | 5332 | 1.8112 | 1.3545 | 4.2529 | 0.9073 | 1.7126 | 0.9178 |
| state_nll | drone | 146 | 23.0274 | 23.8186 | 25.6786 | — | 11.5871 | 1.0000 |
| state_nll | aircraft-airborne (2026-10-04) | 9200 | 8.8948 | 8.3655 | 11.5871 | 0.9999 | 11.5871 | 1.0000 |
| state_nll | aircraft-ground (2026-10-04) | 2638 | 8.6023 | 8.5888 | 11.3787 | 1.0000 | 11.5871 | 1.0000 |
| state_nll | aircraft-airborne (2026-10-03) | 11097 | 9.9305 | 9.6451 | 12.8737 | 0.9994 | 11.5871 | 1.0000 |
| state_nll | aircraft-ground (2026-10-03) | 5332 | 8.7571 | 8.3934 | 12.8286 | 0.9987 | 11.5871 | 1.0000 |
| combined_score | drone | 146 | 7.2226 | 7.6388 | 8.6172 | — | 1.4667 | 1.0000 |
| combined_score | aircraft-airborne (2026-10-04) | 9200 | 0.3256 | 0.2106 | 1.4667 | 0.9999 | 1.4667 | 1.0000 |
| combined_score | aircraft-ground (2026-10-04) | 2638 | -0.1210 | -0.2084 | 1.1500 | 1.0000 | 1.4667 | 1.0000 |
| combined_score | aircraft-airborne (2026-10-03) | 11097 | 0.6757 | 0.5279 | 1.9512 | 0.9994 | 1.4667 | 1.0000 |
| combined_score | aircraft-ground (2026-10-03) | 5332 | 0.0048 | -0.2703 | 1.8983 | 0.9987 | 1.4667 | 1.0000 |

## Plain-language conclusion

State NLL AUROC is 0.99990, compared with transition NLL 0.33624; combined AUROC is 0.99990. 100.0% of drone tokens exceed the10-04 airborne state-NLL p95, versus 9.6% for transition NLL and 100.0% for combined.

Location-conditioned state density flags this route much more consistently than transition NLL, which misses most of its steady flight.

Combined does not improve the AUROC over state density alone in this sample.

The largest mean component excess over airborne aircraft is course_nll: 5.335 nats. Component mean differences (drone minus10-04 airborne): location_height_nll +3.646, speed_nll +5.151, course_nll +5.335.

**This is one drone and one route: a demonstration, not a validated detector.** Neighbouring tokens are correlated and source interpolation differs between simulated drone poses and ADS-B aircraft. 10-04 p95 is a descriptive evaluation reference, not an independently validated deployment threshold.

## Fixed method and provenance

- Histograms fit only 1,007,315 eligible3Hz states from 618 train scenarios (09-28..10-02). Each state uses valid endpoints t−6 and t; intervening frames are not required by the plan. No validation/drone counts enter the density.
- Processed aircraft z and the existing interpolated drone z are already height above166m; no second altitude conversion. Speed/course use past2s displacement only. atan2(0,0)=0 for stationary course.
- Metadata geographic bounds use the M7 fixed-origin tangent-plane conversion.250m grid anchored at west/south ENU bounds; final cells are partial. Exact upper bounds are included in the final cells; all other out-of-bounds states use one separate outside bucket.
- Pool raw joint and conditional counts over3×3 spatial neighbours at fixed height with zero padding; outside pools only itself. Add alpha0.5 to every bin. Joint normalises over all(cell,height) including outside; speed/course normalise conditionally.
- State NLL = location_height_nll + speed_nll + course_nll. Height boundaries15/100/300/600m; speed boundaries5/10/20/…/120m/s (14bins);12course bins begin at−pi.
- Standardisation fits only 16,42910-03 aircraft tokens, airborne+ground. Each score uses its own median/IQR; combined=max(z_transition,z_state). Parameters are in standardization.json. A zero IQR raises an error; no fallback/tuning.
- Scores join the exact M8 token rows at token_start_frame. State is causal at the start; transition uses the following2s GT and is available at token end. Combined is not an online score at token start.
- AUROC treats drone as positive and higher scores as anomalous, with average ranks for ties. Exceedance is strict > the separately computed10-04-airborne p95 of each metric.
- Optional combined-score GIFs use that same10-04 airborne combined p95 for the red outline; the labels show combined z-units, not NLL nats. State is known at token start, but the combined label appears only once its transition token completes.
- All bin sizes, smoothing and combination rules were fixed by the plan before evaluating the drone. No NN training/model changes, package installs or drone tuning.

Detailed counts/probabilities: state_density.npz; all token joins: token_scores.csv; run checks: acceptance.json.
