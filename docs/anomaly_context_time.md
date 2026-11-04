# Clock-conditioned grid anomaly experiment

Condition the existing250m grid/height, speed and course density on Chicago local clock07:00–19:00 versus19:00–07:00 (America/Chicago; DST conversion). These are clock periods, not solar day/night or weather categories. UTC = preserved scenario start + frame/3. Audited all618train metadata/index/processed starts and the first calculated state's timestamp/bracketing source times. Original irregular ADS-B samples are documented in the read-only source dataset. No external weather/time verification or completeness claim.

Fit only existing09-28..10-02train states at valid endpoints2s apart, unchanged bins, spatial pooling and speed/course kernels. Global model reproduces frozen M8b probabilities and all28413token state scores exactly. Conditional joint probabilities use1000global-prior pseudo-states; local speed/course conditionals use100global-prior pseudo-states. Fixed before evaluation; not tuned. Unsupported local speed/course conditionals fall back to global; a wholly absent context reproduces global. Location probability uses a globally normalized prior, not per-cell exact fallback; sparse location support may still produce large differences. Pooled3Hz counts are correlated and are not effective independent sample sizes.

Train context states: {'day_07_19': 672638, 'night_19_07': 334677}. All618train scenarios are used; start periods388day/230night. Evaluate the same existing M8b token rows, using actual aircraft timestamps. The single synthetic drone is tagged by its scripted source interval (18:00UTC/13:00Chicago), not an observed real-world drone recording. Counterfactual day/night scoring changes only the context of the identical state, not its coordinates or trajectory.

Calibrate separate global/context p95 cutoffs on10-03AIRBORNE tokens only (each common across contexts);10-04is report-only. Existing demo cutoff and scores stay unchanged. Aircraft flag fraction is a reference-data exceedance rate, not verified threat false positives. AUROC compares one scripted drone route against aircraft, not evidence of real-world threat detection; night-subgroup AUROC compares daytime drone positives against night aircraft and is not a matched-context metric. No uncertainty/significance claims; inspected development data, not a pristine final test.

Source dataset is sampled traffic, some cached daily-history subsets; coverage varies by hour and may confound context with flow/operations or selection. Only5train days and one drone route. Conditional distributions describe the provided sample; they cannot establish that nighttime is inherently suspicious, or that weather/time changes intent. No event-rate/traffic-count model or threat-rule modification.

# Results

10-04airborne token AUROC: global 0.99976 → conditional 0.99968.
Mean state NLL: 9.2493 → 8.9955.
Fraction exceeding their respective10-03p95 cutoffs: 2.00% → 1.29%.
Drone fraction exceeding cutoff: 100.00% → 100.00%.
Same10-04aircraft states, night-minus-day NLL p10/median/p90: [-1.0494623034489106, 0.8920901659122755, 1.49003549052453]. 382/11838change binary anomaly flag when context alone changes.

The context effect is learned and bidirectional: whether the score increases or decreases depends on the state and the observed clock-conditioned distribution. Accuracy need not improve. This is an isolated experiment; no deployed detector or demo changed.

# Outputs

context_density.png: train hourly coverage and learned day/night location ratio. same_state_context_effect.png: identical-state score differences. paired_examples.png/csv: fixed10th/90th-quantile examples from10-03with>=100pooled states in BOTH contexts, illustrative and explicitly selected, not10-04performance. token_scores.csv: actual-clock/global and both counterfactual component scores. coverage.csv/timestamp_audit.csv/state_density.npz/summary.csv/acceptance.json support reproduction. Run: python -m scripts.anomaly_context_time. No model execution, packages, network, weather input or demo edits.

Support check: of 11,547 10-04 aircraft token states with at least 100 pooled train states in EACH context, 205 change flag under a clock-only change; with at least 1,000 each, 106 / 10,022 change. These strata diagnose data support; they were not used to tune the model. Ground-token exceedance on 10-04 also decreases (0.61% to 0.49%), while 10-03 ground exceedance increases (5.57% to 6.53%); improvements are not uniform.
