# Selected clock-context anomaly flips

Showcase illustrations selected for crossing the existing cutoff. These are not randomly sampled examples or new performance measurements. No model, cutoff, score formula, context bins or earlier demo was changed. Map/track geometry is identical between each pair; lighting is not simulated. Clock labels mean Chicago local periods07–19/19–07, not actual solar illumination.

The new synthetic drone is a10s straight constant-speed trajectory constructed to illustrate the frozen aircraft-reference density. Height30m/speed25m/s were selected from a declared finite candidate grid. The original native drone route was not modified. A green NORMAL label means this kinematic state is within the aircraft-reference range; object identity, permission, safety and threat are not inferred.

| Example | Day score | Night score | Cutoff | Main contribution to increase |
|---|---:|---:|---:|---|
| synthetic_drone | 11.80 | 15.50 | 12.64 | Speed at this location |
| reference_aircraft_01 | 11.19 | 14.18 | 12.64 | Direction at this location |
| reference_aircraft_02 | 11.30 | 13.89 | 12.64 | Direction at this location |

Train support counts are spatially pooled and serially correlated. Synthetic current-state support is nonzero in both contexts; the context differences reflect the supplied sample and may include runway flow/selection effects. Two additional cases retain actual10-04reference aircraft histories. Source IDs, selection rules, scores, factors and support are in examples.json. PNGs are3200×1600. Raw tracks are saved as NPZ. No weather, neural inference, training, packages, network or threat changes.
