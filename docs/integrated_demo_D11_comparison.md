# D11 checkpoint comparison

Only the forecast checkpoint changed: epoch016 → M10 epoch018, with 16 runway-end anchors. Same seed, observations, noise, frozen density, cutoff, routes, zone, rules, 32 samples, top-k 5 and temperature 1. No tuning or re-training.

| Scenario | First Potential intrusion, old (s) | Anchor (s) | D4 pre-entry high lead, old (s) | Anchor (s) | Background paired sample change (m) |
|---|---:|---:|---:|---:|---:|
| A | None | None | None | None | 69.61 |
| B | 56.0 | 58.0 | 24.66666666666667 | 22.66666666666667 | 70.91 |

`None` means no qualifying warning/entry. D4 gate uses its original numeric rule with 10 s anomaly memory; displayed Potential intrusion uses the accepted current-only D10 rule. These are reported separately in acceptance.json. Warning lead is relative to reference entry (B: 80.6667 s), not observed noisy entry. Persistent High threat is the observed D9 incident latch, not the forecast warning.

Background change is mean 3D distance between corresponding sample indices across forecast origins, agents fixed to the background aircraft; it describes changes, not accuracy or a coupling of calibrated futures. Full per-object zone-entry shares at every 1/3 s are in zone_entry_over_time.csv; foreground traces appear in zone_entry_over_time.png. Forecasts refresh every 2 s; intervening values assess remaining sampled paths.

Assertions passed exactly: all observation/reference arrays and validity/times/IDs, all current score arrays/components/explanations, cutoffs, per-object orange decisions, and per-frame observed incident flags/IDs/first detection times. All original protected files remain SHA256-identical. Forecasts, medoids, groups, shares and forecast-derived scenario status may change. D10 readable_status renderer and map extents are retained. Interactive D9 viewer is unchanged.

Sample shares are fractions of 32 futures, not calibrated probabilities. State density has location/height, speed and course context; time-of-day and noise are scripted, not learned detector context.
