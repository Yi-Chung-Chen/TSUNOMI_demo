# Full-dataset statistics

All 800 scenarios were ingested without interpolation or sanitization. Exact counts reproduce the index. Some extrema differ from the plan’s preliminary measurements: maximum horizontal speed 433.35 m/s; height -97.42 to 2790.84 m; vertical speed -327.26 to 185.40 m/s. Median speed is 67.50 m/s; ground fraction is 31.37%, altitude-fallback fraction is 34.32%. M3 applies the configured vertical-speed segment filter.

```json
{
  "scenarios": 800,
  "aircraft_tracks": 4967,
  "valid_rows": 1231609,
  "index_reference_samples": 1231609,
  "scenarios_by_date": {
    "2026-09-28": 188,
    "2026-09-29": 110,
    "2026-09-30": 106,
    "2026-10-01": 108,
    "2026-10-02": 106,
    "2026-10-03": 91,
    "2026-10-04": 91
  },
  "aircraft_per_scenario_quantiles": {
    "0": 1.0,
    "5": 1.0,
    "25": 3.0,
    "50": 6.0,
    "75": 9.0,
    "95": 13.0,
    "100": 17.0
  },
  "type_histogram": {
    "E75L": 865,
    "CRJ7": 782,
    "B738": 487,
    "E170": 453,
    "A319": 422,
    "CRJ2": 309,
    "A21N": 268,
    "A320": 179,
    "B739": 168,
    "B737": 124,
    "B38M": 123,
    "": 117,
    "A321": 102,
    "B39M": 78,
    "B77L": 49,
    "B788": 44,
    "BCS3": 38,
    "B744": 33,
    "B78X": 30,
    "B789": 27,
    "A333": 24,
    "B763": 23,
    "B77W": 20,
    "B748": 20,
    "BCS1": 19,
    "E145": 16,
    "CRJ9": 15,
    "B712": 15,
    "B753": 12,
    "DH8D": 10,
    "A35K": 9,
    "B772": 9,
    "A359": 9,
    "E135": 8,
    "B752": 6,
    "C68A": 6,
    "CL35": 5,
    "E75S": 5,
    "A20N": 5,
    "A332": 5,
    "E545": 4,
    "E55P": 4,
    "C25B": 3,
    "C56X": 3,
    "C17": 2,
    "AS55": 2,
    "G150": 1,
    "A343": 1,
    "B429": 1,
    "FA20": 1,
    "GLF5": 1,
    "CL30": 1,
    "GLF4": 1,
    "A339": 1,
    "H25B": 1,
    "F2TH": 1
  },
  "ground_fraction": 0.31367909783056147,
  "altitude_fallback_fraction": 0.3432038901956709,
  "horizontal_speed_mps_quantiles": {
    "0": 0.0,
    "5": 2.4491739040813827,
    "25": 18.9207699276281,
    "50": 67.49854030620287,
    "75": 79.43112796598399,
    "95": 95.62589205743335,
    "100": 433.34580840670225
  },
  "height_m_quantiles": {
    "0": -97.41999816894531,
    "5": 0.800000011920929,
    "25": 0.800000011920929,
    "50": 59.124210357666016,
    "75": 204.01173400878906,
    "95": 562.0877685546872,
    "100": 2790.837158203125
  },
  "vertical_speed_mps_quantiles": {
    "0": -327.2589912414551,
    "5": -7.493019104003906,
    "25": 0.0,
    "50": 0.0,
    "75": 7.10009765625,
    "95": 16.565185546875,
    "100": 185.3999953866005
  },
  "position_min_m": [
    -3824.86279296875,
    -4454.82177734375,
    -97.41999816894531
  ],
  "position_max_m": [
    4881.77197265625,
    3878.217041015625,
    2790.837158203125
  ]
}
```
