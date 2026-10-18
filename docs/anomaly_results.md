# Anomaly scoring MVP

Higher NLL/residual = more anomalous; AUROC treats drone tokens as positives.

| group | token_count | nll_mean | nll_median | nll_p95 | residual_horizontal_m_mean | residual_horizontal_m_p95 | residual_vertical_m_mean | residual_vertical_m_p95 | nll_auroc | residual_horizontal_m_auroc | residual_vertical_m_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drone | 146 | 3.3924 | 2.6578 | 7.8579 | 1.7017 | 2.9596 | 0.2916 | 1.1784 | — | — | — |
| aircraft-airborne (2026-10-04) | 9200 | 4.0679 | 3.8981 | 7.0834 | 2.2138 | 4.6906 | 1.4322 | 3.5086 | 0.3362 | 0.3998 | 0.1114 |
| aircraft-ground (2026-10-04) | 2638 | 2.0781 | 1.4581 | 5.7732 | 1.1975 | 2.4980 | 0.1101 | 0.6909 | 0.7597 | 0.7048 | 0.8698 |
| aircraft-airborne (2026-10-03) | 11097 | 4.4526 | 4.3127 | 7.5434 | 2.1780 | 4.7812 | 1.4468 | 3.6998 | 0.2829 | 0.4173 | 0.1303 |
| aircraft-ground (2026-10-03) | 5332 | 2.1235 | 1.5274 | 5.7660 | 1.2755 | 2.6918 | 0.1755 | 0.7990 | 0.7467 | 0.6847 | 0.8363 |

## Plain-language result

The drone has median NLL 2.658, versus 3.898 for 10-04 airborne aircraft. NLL AUROC is 0.336 (0.5 is chance; 1 is perfect separation with higher drone scores). The drone is generally assigned higher likelihood than airborne aircraft, so treating high NLL as an anomaly does not identify this drone reliably.

The drone track-mean NLL is at the 16.7th percentile of 354 10-04 airborne aircraft track-segment means (each scenario/agent pair is one segment). This is a descriptive rank, not an extra score.

The drone reconstructs with mean horizontal / vertical residual 1.702 / 0.292 m, versus 2.214 / 1.432 m for airborne aircraft. Residual AUROC is 0.400 horizontally and 0.111 vertically. The drone reconstructs at least as well as aircraft on average in both axes; the aircraft dictionary does not reject this route. These are reconstruction errors, not generated-future forecast errors.

## Method and limits

- Frozen epoch_016.ckpt and existing K2048/shift6 dictionary; eval mode; no training or model changes.
- Each aircraft/drone is an isolated one-agent scene. Batched graphs have distinct scene IDs, so there is no cross-agent attention.
- Re-index 109-frame windows at stride6 over each listed validation scenario, once per (scenario, agent, start). Frames past the scenario end are padded invalid, allowing the last fully observed scored tokens. Existing stride90 demo windows are not reused or duplicated. Only the first future token node3→4 is scored; history0..18 and scored segment18..24 must be valid. No ten-second future eligibility requirement is imposed for this two-second diagnostic.
- NLL uses the default autoregressive GT tokenizer chain and unsmoothed log-softmax, in natural-log units. The decoder is causal: raw future data beyond the scored token is not input to its prediction.
- Residual uses a separate nearest-token match from the raw anchor at frame18, with the existing endpoint contour + vertical matching metric and causal chord headings. Report mean pointwise xy distance / absolute z error over frames19..24, in metres. Its token ID can differ from the chain label; accumulated chain drift is excluded.
- Drone native poses are linearly interpolated at 0..900 / 3 s; heading unwraps before interpolation. Unsupported boundary times are invalid, never extrapolated. Native-knot reproduction tests the interpolant at its original knots, not a lossy grid→native round-trip.
- The native file has902 reads at901 unique timestamps: reads195/196 have identical time65.49s, xyz and heading. They are collapsed for interpolation without changing any raw file; conflicting duplicates raise an error.
- Drone height is native ENU z +2250−166 m; all drone samples are airborne, type UNK. No camera tracks.
- Native drone heading vs movement median absolute difference is 0.024 degrees; its CCW-from-east convention is consistent and unchanged. Two-sided interpolation can use a bracketing read about0.33s later, as aircraft interpolation also uses both sides.
- 10-04 is the primary normal comparison; 10-03 selected the checkpoint/dictionary settings and is reference only.
- Tokens overlap in history, and all drone positives come from one simulated route. Counts are tokens, not independent flights; AUROC is descriptive and no operating threshold or general anomaly-detection claim is established.
- Drone poses are smooth native Unreal reads near3Hz, whereas aircraft positions are piecewise-linear interpolation of ADS-B near1Hz. This acquisition/interpolation difference can shift residual and NLL independently of vehicle identity.
- Type/state/domain differences remain even with identical single-agent context. An autoregressive transition score can reward steady motion despite an unfamiliar state. No state-plausibility or fallback score was added.

Run receipt is in acceptance.json; individual scores are in token_scores.csv.
