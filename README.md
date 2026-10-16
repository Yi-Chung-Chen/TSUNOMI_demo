# sygard-motion

Aircraft trajectory prediction on the 800-scenario SYGARD ORD dataset. Work follows `../SYGARD_MOTION_PLAN.md`. Raw data stays outside this repository; all processed outputs are ignored.

Use the existing `catk` environment through `scripts/catk.sh`. Training uses only physical GPUs 1 and 3 after verifying availability. No remote is configured.

Timing: 3 Hz, 6 s history, 2 s tokens, 30 s future. Split by UTC date; flights may continue across midnight at split boundaries. Heights use altitude minus 166 m and retain altitude fallback flags.

Code attribution: copied SMART/CAT-K files retain their original headers; see LICENSE and LICENSE.SMART. Raw data originates from ADSB.lol historical traces (ODbL 1.0); see the source dataset README for provenance.
