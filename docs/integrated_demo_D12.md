# Integrated demo v7 — tracker presentation

Read frozen D11 anchor forecasts only. Recent 10 seconds of raw tracker history fade from dim to bright over a faint full-history line. A fixed-duration tail is longer at higher speed. Original tracker noise remains visible.

Icons use a causal linear fit over the last 7 observed samples (2 s), evaluated at the current time. Orientation uses the latest model joint-medoid tangent over the next 2 s, circularly eased with a 0.6 s time constant. Before the first forecast at 6 s, heading uses filtered observed displacement. These are display transforms only.

Raw observations, all32 forecasts, sampled groups/percentages, anomalies, explanations, threat timelines and incident memory are unchanged. Map extents and cards retain D11 settings. B still warns at58s (22.67s before reference entry), and A has no intrusion. No inference, training, packages, network or new scenarios. Previous versions and interactive viewer are preserved.

MP4/GIF:1600x900,3x replay. Review PNG:3200x1800. Compare D11/D12 at A62.33s and B56/58/63.33s. Additional PNGs show before/after acceleration and post-intrusion memory.
