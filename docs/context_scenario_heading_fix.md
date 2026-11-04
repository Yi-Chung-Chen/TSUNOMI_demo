# Heading initialization repair

Heading is unknown until an observed displacement or already-issued forecast supplies a valid direction. The first valid direction initializes directly; subsequent updates retain circular easing. Unknown heading uses a small nondirectional dot instead of an east-facing aircraft/drone icon. No future observations or reference/GT heading are used.

Re-rendered day/night scenario media into outputs/context_scenario_demo_heading_fixed, preserving original outputs/context_scenario_demo. Render-only loads saved observations/forecasts; zero new inference. Timelines and future groups match the old JSON byte-for-byte; all anomaly scores, cutoff, status, inputs, forecasts, map bounds and threat/latch rules remain unchanged. Example aircraft now initializes directly to165.82degrees at0.33s, instead of easing from a fabricated0degrees. No initialization rotation.

21 targeted tests passed; both MP4s decode completely and all-frame layout checks pass. Review frames0/1/3/6/18/30 saved. Reproduce with python -m scripts.context_scenario_demo --render-only. Check acceptance JSON for numeric/protection evidence.
