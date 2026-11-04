# Initialize visible heading from supplied history

The first6s of existing observed tracks are history, not playback startup. The first visible frame is t=6s and initializes each icon directly from its past6s filtered displacement. Aircraft and drone icons are present from the first visible frame; no circle or assumed east heading. Later poses ease toward already-issued forecasts. No future or reference/GT direction is used.

Saved source is10s, so visible motion is6..10s (4s), with full preceding history shown. No artificial prehistory, route extension or new inference was introduced to lengthen the clip. Model inputs, saved forecasts, detector cutoff and full source timelines/groups are unchanged, byte-for-byte against original JSON. Earlier media remain available.

Corrected MP4/GIF/review frames: outputs/context_scenario_demo_history_heading/{daytime,nighttime}. Reproduce: python -m scripts.context_scenario_demo --render-only.23targeted tests pass, including causal history initialization despite conflicting forecast, and directional icons on the first visible frame. Acceptance JSON records original/visible frame counts and heading values.
