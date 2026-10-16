# BC Demo evaluation

BC checkpoint: epoch_016.ckpt; no retraining. Val merges existing windows: 2026-10-03 (488) and 2026-10-04 (424). 10-03 was used to select both checkpoint epoch16 and dictionary configuration K2048/chord heading; dictionary centres were built from train only. 10-04 was not used for either choice. Combined results are not an untouched test score.

32 joint rollouts, topk_prob K=5, temperature=1.0. ADE/FDE average over rollouts; minADE is the per-agent minimum whole-trajectory ADE. Horizontal/vertical minima are independent. Miss rate = all 32 horizontal endpoints exceed 500 m at 30 s; mean_rollout_miss_rate is also reported. Targets need continuous GT through each horizon. Counts are agent-window instances; aggregation weights agents, not windows.

## Does BC beat constant velocity?

Lower is better. This compares minADE, ADE and FDE separately; each comparison is paired on the same eligible agents.

| Date | Group | Horizon s | Axis | Agents | minADE: BC / CV / BC wins | ADE: BC / CV / BC wins | FDE: BC / CV / BC wins |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | airborne | 10 | horizontal | 602 | 12.462 / 36.797 / yes | 22.892 / 36.797 / yes | 52.178 / 90.005 / yes |
| 2026-10-03 | airborne | 10 | vertical | 602 | 4.445 / 16.290 / yes | 9.067 / 16.290 / yes | 16.609 / 33.576 / yes |
| 2026-10-03 | airborne | 20 | horizontal | 519 | 33.790 / 110.386 / yes | 62.908 / 110.386 / yes | 167.963 / 295.492 / yes |
| 2026-10-03 | airborne | 20 | vertical | 519 | 8.017 / 37.293 / yes | 17.628 / 37.293 / yes | 35.258 / 80.357 / yes |
| 2026-10-03 | airborne | 30 | horizontal | 406 | 69.157 / 210.121 / yes | 124.456 / 210.121 / yes | 365.227 / 590.499 / yes |
| 2026-10-03 | airborne | 30 | vertical | 406 | 11.822 / 63.365 / yes | 26.398 / 63.365 / yes | 52.115 / 140.220 / yes |
| 2026-10-03 | all | 10 | horizontal | 920 | 10.594 / 29.176 / yes | 21.412 / 29.176 / yes | 48.705 / 73.180 / yes |
| 2026-10-03 | all | 10 | vertical | 920 | 3.297 / 11.446 / yes | 6.592 / 11.446 / yes | 12.163 / 24.018 / yes |
| 2026-10-03 | all | 20 | horizontal | 807 | 28.379 / 91.488 / yes | 59.146 / 91.488 / yes | 156.083 / 250.172 / yes |
| 2026-10-03 | all | 20 | vertical | 807 | 6.265 / 26.816 / yes | 13.171 / 26.816 / yes | 27.325 / 59.419 / yes |
| 2026-10-03 | all | 30 | horizontal | 668 | 57.051 / 180.644 / yes | 118.225 / 180.644 / yes | 337.635 / 516.346 / yes |
| 2026-10-03 | all | 30 | vertical | 668 | 10.236 / 45.607 / yes | 20.762 / 45.607 / yes | 46.079 / 106.136 / yes |
| 2026-10-03 | ground | 10 | horizontal | 318 | 7.055 / 14.748 / yes | 18.610 / 14.748 / no | 42.130 / 41.328 / no |
| 2026-10-03 | ground | 10 | vertical | 318 | 1.124 / 2.277 / yes | 1.907 / 2.277 / yes | 3.745 / 5.923 / yes |
| 2026-10-03 | ground | 20 | horizontal | 288 | 18.629 / 57.432 / yes | 52.366 / 57.432 / yes | 134.675 / 168.500 / yes |
| 2026-10-03 | ground | 20 | vertical | 288 | 3.107 / 7.936 / yes | 5.140 / 7.936 / yes | 13.028 / 21.688 / yes |
| 2026-10-03 | ground | 30 | horizontal | 262 | 38.290 / 134.964 / yes | 108.569 / 134.964 / yes | 294.879 / 401.436 / yes |
| 2026-10-03 | ground | 30 | vertical | 262 | 7.778 / 18.088 / yes | 12.030 / 18.088 / yes | 36.725 / 53.318 / yes |
| 2026-10-04 | airborne | 10 | horizontal | 476 | 9.080 / 30.413 / yes | 16.592 / 30.413 / yes | 35.066 / 65.914 / yes |
| 2026-10-04 | airborne | 10 | vertical | 476 | 3.566 / 17.984 / yes | 6.713 / 17.984 / yes | 11.570 / 35.646 / yes |
| 2026-10-04 | airborne | 20 | horizontal | 396 | 20.048 / 71.723 / yes | 39.675 / 71.723 / yes | 96.274 / 171.034 / yes |
| 2026-10-04 | airborne | 20 | vertical | 396 | 5.771 / 37.173 / yes | 11.458 / 37.173 / yes | 21.301 / 77.459 / yes |
| 2026-10-04 | airborne | 30 | horizontal | 318 | 36.480 / 124.652 / yes | 71.617 / 124.652 / yes | 191.465 / 318.642 / yes |
| 2026-10-04 | airborne | 30 | vertical | 318 | 7.083 / 60.850 / yes | 15.171 / 60.850 / yes | 27.727 / 128.764 / yes |
| 2026-10-04 | all | 10 | horizontal | 620 | 8.841 / 26.895 / yes | 16.857 / 26.895 / yes | 36.281 / 60.559 / yes |
| 2026-10-04 | all | 10 | vertical | 620 | 2.958 / 14.238 / yes | 5.483 / 14.238 / yes | 9.577 / 28.353 / yes |
| 2026-10-04 | all | 20 | horizontal | 514 | 20.850 / 70.181 / yes | 43.508 / 70.181 / yes | 107.849 / 173.803 / yes |
| 2026-10-04 | all | 20 | vertical | 514 | 5.109 / 29.986 / yes | 9.789 / 29.986 / yes | 19.063 / 63.244 / yes |
| 2026-10-04 | all | 30 | horizontal | 422 | 39.486 / 131.152 / yes | 84.006 / 131.152 / yes | 225.989 / 345.054 / yes |
| 2026-10-04 | all | 30 | vertical | 422 | 7.258 / 49.308 / yes | 14.023 / 49.308 / yes | 28.890 / 107.278 / yes |
| 2026-10-04 | ground | 10 | horizontal | 144 | 8.051 / 15.265 / yes | 17.732 / 15.265 / no | 40.296 / 42.858 / yes |
| 2026-10-04 | ground | 10 | vertical | 144 | 0.945 / 1.858 / yes | 1.417 / 1.858 / yes | 2.989 / 4.247 / yes |
| 2026-10-04 | ground | 20 | horizontal | 118 | 23.542 / 65.004 / yes | 56.372 / 65.004 / yes | 146.697 / 183.094 / yes |
| 2026-10-04 | ground | 20 | vertical | 118 | 2.888 / 5.867 / yes | 4.191 / 5.867 / yes | 11.550 / 15.539 / yes |
| 2026-10-04 | ground | 30 | horizontal | 104 | 48.678 / 151.029 / yes | 121.888 / 151.029 / yes | 331.552 / 425.812 / yes |
| 2026-10-04 | ground | 30 | vertical | 104 | 7.796 / 14.017 / yes | 10.516 / 14.017 / yes | 32.446 / 41.580 / yes |
| combined | airborne | 10 | horizontal | 1078 | 10.969 / 33.978 / yes | 20.110 / 33.978 / yes | 44.622 / 79.367 / yes |
| combined | airborne | 10 | vertical | 1078 | 4.057 / 17.038 / yes | 8.027 / 17.038 / yes | 14.384 / 34.490 / yes |
| combined | airborne | 20 | horizontal | 915 | 27.842 / 93.653 / yes | 52.853 / 93.653 / yes | 136.937 / 241.628 / yes |
| combined | airborne | 20 | vertical | 915 | 7.045 / 37.241 / yes | 14.957 / 37.241 / yes | 29.218 / 79.102 / yes |
| combined | airborne | 30 | horizontal | 724 | 54.804 / 172.581 / yes | 101.248 / 172.581 / yes | 288.906 / 471.093 / yes |
| combined | airborne | 30 | vertical | 724 | 9.740 / 62.260 / yes | 21.466 / 62.260 / yes | 41.403 / 135.188 / yes |
| combined | all | 10 | horizontal | 1540 | 9.888 / 28.257 / yes | 19.578 / 28.257 / yes | 43.703 / 68.099 / yes |
| combined | all | 10 | vertical | 1540 | 3.160 / 12.570 / yes | 6.146 / 12.570 / yes | 11.122 / 25.763 / yes |
| combined | all | 20 | horizontal | 1321 | 25.450 / 83.197 / yes | 53.061 / 83.197 / yes | 137.316 / 220.456 / yes |
| combined | all | 20 | vertical | 1321 | 5.815 / 28.050 / yes | 11.855 / 28.050 / yes | 24.110 / 60.907 / yes |
| combined | all | 30 | horizontal | 1090 | 50.250 / 161.483 / yes | 104.977 / 161.483 / yes | 294.411 / 450.029 / yes |
| combined | all | 30 | vertical | 1090 | 9.083 / 47.040 / yes | 18.153 / 47.040 / yes | 39.424 / 106.578 / yes |
| combined | ground | 10 | horizontal | 462 | 7.366 / 14.909 / yes | 18.336 / 14.909 / no | 41.559 / 41.805 / yes |
| combined | ground | 10 | vertical | 462 | 1.069 / 2.147 / yes | 1.754 / 2.147 / yes | 3.510 / 5.400 / yes |
| combined | ground | 20 | horizontal | 406 | 20.057 / 59.633 / yes | 53.530 / 59.633 / yes | 138.169 / 172.741 / yes |
| combined | ground | 20 | vertical | 406 | 3.044 / 7.335 / yes | 4.864 / 7.335 / yes | 12.598 / 19.901 / yes |
| combined | ground | 30 | horizontal | 366 | 41.242 / 139.529 / yes | 112.353 / 139.529 / yes | 305.300 / 408.362 / yes |
| combined | ground | 30 | vertical | 366 | 7.783 / 16.931 / yes | 11.600 / 16.931 / yes | 35.509 / 49.983 / yes |

## All metrics

| Model | Date | Group | Horizon s | Axis | Agents | minADE m | ADE m | FDE m | Miss rate | Mean rollout miss rate |
|---|---|---|---|---|---|---|---|---|---|---|
| bc | 2026-10-03 | airborne | 10 | horizontal | 602 | 12.462487 | 22.891954 | 52.178442 | NA | NA |
| bc | 2026-10-03 | airborne | 10 | vertical | 602 | 4.444874 | 9.067072 | 16.609242 | NA | NA |
| bc | 2026-10-03 | airborne | 20 | horizontal | 519 | 33.789573 | 62.907703 | 167.963297 | NA | NA |
| bc | 2026-10-03 | airborne | 20 | vertical | 519 | 8.017254 | 17.627870 | 35.258158 | NA | NA |
| bc | 2026-10-03 | airborne | 30 | horizontal | 406 | 69.157076 | 124.456413 | 365.226579 | 0.103448 | 0.240148 |
| bc | 2026-10-03 | airborne | 30 | vertical | 406 | 11.821931 | 26.397589 | 52.115073 | NA | NA |
| bc | 2026-10-03 | all | 10 | horizontal | 920 | 10.593516 | 21.411800 | 48.705289 | NA | NA |
| bc | 2026-10-03 | all | 10 | vertical | 920 | 3.297167 | 6.592180 | 12.162817 | NA | NA |
| bc | 2026-10-03 | all | 20 | horizontal | 807 | 28.379214 | 59.145596 | 156.083426 | NA | NA |
| bc | 2026-10-03 | all | 20 | vertical | 807 | 6.265002 | 13.171335 | 27.324590 | NA | NA |
| bc | 2026-10-03 | all | 30 | horizontal | 668 | 57.050585 | 118.224951 | 337.635322 | 0.070359 | 0.224925 |
| bc | 2026-10-03 | all | 30 | vertical | 668 | 10.235857 | 20.762482 | 46.078709 | NA | NA |
| bc | 2026-10-03 | ground | 10 | horizontal | 318 | 7.055401 | 18.609749 | 42.130326 | NA | NA |
| bc | 2026-10-03 | ground | 10 | vertical | 318 | 1.124466 | 1.907005 | 3.745373 | NA | NA |
| bc | 2026-10-03 | ground | 20 | horizontal | 288 | 18.629295 | 52.365967 | 134.674909 | NA | NA |
| bc | 2026-10-03 | ground | 20 | vertical | 288 | 3.107297 | 5.140287 | 13.027638 | NA | NA |
| bc | 2026-10-03 | ground | 30 | horizontal | 262 | 38.290146 | 108.568564 | 294.879406 | 0.019084 | 0.201336 |
| bc | 2026-10-03 | ground | 30 | vertical | 262 | 7.778048 | 12.030216 | 36.724650 | NA | NA |
| bc | 2026-10-04 | airborne | 10 | horizontal | 476 | 9.079601 | 16.592432 | 35.065878 | NA | NA |
| bc | 2026-10-04 | airborne | 10 | vertical | 476 | 3.566477 | 6.712557 | 11.569578 | NA | NA |
| bc | 2026-10-04 | airborne | 20 | horizontal | 396 | 20.048069 | 39.674826 | 96.273665 | NA | NA |
| bc | 2026-10-04 | airborne | 20 | vertical | 396 | 5.771109 | 11.457555 | 21.301012 | NA | NA |
| bc | 2026-10-04 | airborne | 30 | horizontal | 318 | 36.479909 | 71.616967 | 191.464769 | 0.037736 | 0.106034 |
| bc | 2026-10-04 | airborne | 30 | vertical | 318 | 7.082596 | 15.170543 | 27.726955 | NA | NA |
| bc | 2026-10-04 | all | 10 | horizontal | 620 | 8.840789 | 16.857222 | 36.280698 | NA | NA |
| bc | 2026-10-04 | all | 10 | vertical | 620 | 2.957693 | 5.482691 | 9.576602 | NA | NA |
| bc | 2026-10-04 | all | 20 | horizontal | 514 | 20.850264 | 43.508143 | 107.849492 | NA | NA |
| bc | 2026-10-04 | all | 20 | vertical | 514 | 5.109239 | 9.789336 | 19.062529 | NA | NA |
| bc | 2026-10-04 | all | 30 | horizontal | 422 | 39.486025 | 84.005954 | 225.988630 | 0.045024 | 0.132998 |
| bc | 2026-10-04 | all | 30 | vertical | 422 | 7.258359 | 14.023404 | 28.890035 | NA | NA |
| bc | 2026-10-04 | ground | 10 | horizontal | 144 | 8.051383 | 17.732498 | 40.296353 | NA | NA |
| bc | 2026-10-04 | ground | 10 | vertical | 144 | 0.945323 | 1.417300 | 2.988711 | NA | NA |
| bc | 2026-10-04 | ground | 20 | horizontal | 118 | 23.542375 | 56.372496 | 146.697185 | NA | NA |
| bc | 2026-10-04 | ground | 20 | vertical | 118 | 2.888048 | 4.190905 | 11.550331 | NA | NA |
| bc | 2026-10-04 | ground | 30 | horizontal | 104 | 48.677803 | 121.887666 | 331.551974 | 0.067308 | 0.215445 |
| bc | 2026-10-04 | ground | 30 | vertical | 104 | 7.795789 | 10.515807 | 32.446373 | NA | NA |
| bc | combined | airborne | 10 | horizontal | 1078 | 10.968745 | 20.110347 | 44.622245 | NA | NA |
| bc | combined | airborne | 10 | vertical | 1078 | 4.057010 | 8.027416 | 14.383936 | NA | NA |
| bc | combined | airborne | 20 | horizontal | 915 | 27.842430 | 52.852818 | 136.936964 | NA | NA |
| bc | combined | airborne | 20 | vertical | 915 | 7.045152 | 14.957439 | 29.217688 | NA | NA |
| bc | combined | airborne | 30 | horizontal | 724 | 54.804398 | 101.247927 | 288.905784 | 0.074586 | 0.181241 |
| bc | combined | airborne | 30 | vertical | 724 | 9.740289 | 21.466372 | 41.403165 | NA | NA |
| bc | combined | all | 10 | horizontal | 1540 | 9.887872 | 19.578139 | 43.703181 | NA | NA |
| bc | combined | all | 10 | vertical | 1540 | 3.160496 | 6.145502 | 11.121614 | NA | NA |
| bc | combined | all | 20 | horizontal | 1321 | 25.449706 | 53.061076 | 137.315643 | NA | NA |
| bc | combined | all | 20 | vertical | 1321 | 5.815296 | 11.855402 | 24.109829 | NA | NA |
| bc | combined | all | 30 | horizontal | 1090 | 50.250361 | 104.976863 | 294.410640 | 0.060550 | 0.189335 |
| bc | combined | all | 30 | vertical | 1090 | 9.083101 | 18.153408 | 39.424011 | NA | NA |
| bc | combined | ground | 10 | horizontal | 462 | 7.365837 | 18.336320 | 41.558698 | NA | NA |
| bc | combined | ground | 10 | vertical | 462 | 1.068629 | 1.754370 | 3.509530 | NA | NA |
| bc | combined | ground | 20 | horizontal | 406 | 20.057235 | 53.530426 | 138.169068 | NA | NA |
| bc | combined | ground | 20 | vertical | 406 | 3.043574 | 4.864358 | 12.598273 | NA | NA |
| bc | combined | ground | 30 | horizontal | 366 | 41.241830 | 112.353227 | 305.300026 | 0.032787 | 0.205345 |
| bc | combined | ground | 30 | vertical | 366 | 7.783089 | 11.599893 | 35.508964 | NA | NA |
| constant_velocity | 2026-10-03 | airborne | 10 | horizontal | 602 | 36.797241 | 36.797241 | 90.005483 | NA | NA |
| constant_velocity | 2026-10-03 | airborne | 10 | vertical | 602 | 16.289662 | 16.289662 | 33.576189 | NA | NA |
| constant_velocity | 2026-10-03 | airborne | 20 | horizontal | 519 | 110.385913 | 110.385913 | 295.492161 | NA | NA |
| constant_velocity | 2026-10-03 | airborne | 20 | vertical | 519 | 37.293275 | 37.293275 | 80.356561 | NA | NA |
| constant_velocity | 2026-10-03 | airborne | 30 | horizontal | 406 | 210.121374 | 210.121374 | 590.499337 | 0.423645 | 0.423645 |
| constant_velocity | 2026-10-03 | airborne | 30 | vertical | 406 | 63.364926 | 63.364926 | 140.220244 | NA | NA |
| constant_velocity | 2026-10-03 | all | 10 | horizontal | 920 | 29.175720 | 29.175720 | 73.179964 | NA | NA |
| constant_velocity | 2026-10-03 | all | 10 | vertical | 920 | 11.446258 | 11.446258 | 24.017647 | NA | NA |
| constant_velocity | 2026-10-03 | all | 20 | horizontal | 807 | 91.487868 | 91.487868 | 250.171510 | NA | NA |
| constant_velocity | 2026-10-03 | all | 20 | vertical | 807 | 26.816382 | 26.816382 | 59.418957 | NA | NA |
| constant_velocity | 2026-10-03 | all | 30 | horizontal | 668 | 180.643663 | 180.643663 | 516.345628 | 0.396707 | 0.396707 |
| constant_velocity | 2026-10-03 | all | 30 | vertical | 668 | 45.606543 | 45.606543 | 106.135783 | NA | NA |
| constant_velocity | 2026-10-03 | ground | 10 | horizontal | 318 | 14.747559 | 14.747559 | 41.327883 | NA | NA |
| constant_velocity | 2026-10-03 | ground | 10 | vertical | 318 | 2.277298 | 2.277298 | 5.922545 | NA | NA |
| constant_velocity | 2026-10-03 | ground | 20 | horizontal | 288 | 57.432017 | 57.432017 | 168.499919 | NA | NA |
| constant_velocity | 2026-10-03 | ground | 20 | vertical | 288 | 7.936147 | 7.936147 | 21.687650 | NA | NA |
| constant_velocity | 2026-10-03 | ground | 30 | horizontal | 262 | 134.964461 | 134.964461 | 401.435682 | 0.354962 | 0.354962 |
| constant_velocity | 2026-10-03 | ground | 30 | vertical | 262 | 18.087827 | 18.087827 | 53.317877 | NA | NA |
| constant_velocity | 2026-10-04 | airborne | 10 | horizontal | 476 | 30.413204 | 30.413204 | 65.913544 | NA | NA |
| constant_velocity | 2026-10-04 | airborne | 10 | vertical | 476 | 17.983510 | 17.983510 | 35.646015 | NA | NA |
| constant_velocity | 2026-10-04 | airborne | 20 | horizontal | 396 | 71.723364 | 71.723364 | 171.034174 | NA | NA |
| constant_velocity | 2026-10-04 | airborne | 20 | vertical | 396 | 37.172718 | 37.172718 | 77.458764 | NA | NA |
| constant_velocity | 2026-10-04 | airborne | 30 | horizontal | 318 | 124.651772 | 124.651772 | 318.642420 | 0.179245 | 0.179245 |
| constant_velocity | 2026-10-04 | airborne | 30 | vertical | 318 | 60.849710 | 60.849710 | 128.763980 | NA | NA |
| constant_velocity | 2026-10-04 | all | 10 | horizontal | 620 | 26.894904 | 26.894904 | 60.558655 | NA | NA |
| constant_velocity | 2026-10-04 | all | 10 | vertical | 620 | 14.238251 | 14.238251 | 28.353402 | NA | NA |
| constant_velocity | 2026-10-04 | all | 20 | horizontal | 514 | 70.180830 | 70.180830 | 173.802698 | NA | NA |
| constant_velocity | 2026-10-04 | all | 20 | vertical | 514 | 29.985848 | 29.985848 | 63.243741 | NA | NA |
| constant_velocity | 2026-10-04 | all | 30 | horizontal | 422 | 131.152338 | 131.152338 | 345.053896 | 0.218009 | 0.218009 |
| constant_velocity | 2026-10-04 | all | 30 | vertical | 422 | 49.307981 | 49.307981 | 107.277905 | NA | NA |
| constant_velocity | 2026-10-04 | ground | 10 | horizontal | 144 | 15.264967 | 15.264967 | 42.857769 | NA | NA |
| constant_velocity | 2026-10-04 | ground | 10 | vertical | 144 | 1.858091 | 1.858091 | 4.247262 | NA | NA |
| constant_velocity | 2026-10-04 | ground | 20 | horizontal | 118 | 65.004193 | 65.004193 | 183.093678 | NA | NA |
| constant_velocity | 2026-10-04 | ground | 20 | vertical | 118 | 5.867199 | 5.867199 | 15.539086 | NA | NA |
| constant_velocity | 2026-10-04 | ground | 30 | horizontal | 104 | 151.029068 | 151.029068 | 425.812063 | 0.336538 | 0.336538 |
| constant_velocity | 2026-10-04 | ground | 30 | vertical | 104 | 14.016927 | 14.016927 | 41.580100 | NA | NA |
| constant_velocity | combined | airborne | 10 | horizontal | 1078 | 33.978315 | 33.978315 | 79.367484 | NA | NA |
| constant_velocity | combined | airborne | 10 | vertical | 1078 | 17.037594 | 17.037594 | 34.490138 | NA | NA |
| constant_velocity | combined | airborne | 20 | horizontal | 915 | 93.653269 | 93.653269 | 241.628377 | NA | NA |
| constant_velocity | combined | airborne | 20 | vertical | 915 | 37.241100 | 37.241100 | 79.102433 | NA | NA |
| constant_velocity | combined | airborne | 30 | horizontal | 724 | 172.580858 | 172.580858 | 471.092570 | 0.316298 | 0.316298 |
| constant_velocity | combined | airborne | 30 | vertical | 724 | 62.260176 | 62.260176 | 135.188349 | NA | NA |
| constant_velocity | combined | all | 10 | horizontal | 1540 | 28.257469 | 28.257469 | 68.098658 | NA | NA |
| constant_velocity | combined | all | 10 | vertical | 1540 | 12.570307 | 12.570307 | 25.763210 | NA | NA |
| constant_velocity | combined | all | 20 | horizontal | 1321 | 83.197317 | 83.197317 | 220.456469 | NA | NA |
| constant_velocity | combined | all | 20 | vertical | 1321 | 28.049619 | 28.049619 | 60.907177 | NA | NA |
| constant_velocity | combined | all | 30 | horizontal | 1090 | 161.482801 | 161.482801 | 450.029013 | 0.327523 | 0.327523 |
| constant_velocity | combined | all | 30 | vertical | 1090 | 47.039577 | 47.039577 | 106.577962 | NA | NA |
| constant_velocity | combined | ground | 10 | horizontal | 462 | 14.908829 | 14.908829 | 41.804730 | NA | NA |
| constant_velocity | combined | ground | 10 | vertical | 462 | 2.146636 | 2.146636 | 5.400379 | NA | NA |
| constant_velocity | combined | ground | 20 | horizontal | 406 | 59.632797 | 59.632797 | 172.741455 | NA | NA |
| constant_velocity | combined | ground | 20 | vertical | 406 | 7.334828 | 7.334828 | 19.900629 | NA | NA |
| constant_velocity | combined | ground | 30 | horizontal | 366 | 139.529267 | 139.529267 | 408.362304 | 0.349727 | 0.349727 |
| constant_velocity | combined | ground | 30 | vertical | 366 | 16.931069 | 16.931069 | 49.982553 | NA | NA |
| gt_replay | 2026-10-03 | airborne | 10 | horizontal | 602 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | airborne | 10 | vertical | 602 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | airborne | 20 | horizontal | 519 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | airborne | 20 | vertical | 519 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | airborne | 30 | horizontal | 406 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | 2026-10-03 | airborne | 30 | vertical | 406 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | all | 10 | horizontal | 920 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | all | 10 | vertical | 920 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | all | 20 | horizontal | 807 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | all | 20 | vertical | 807 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | all | 30 | horizontal | 668 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | 2026-10-03 | all | 30 | vertical | 668 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | ground | 10 | horizontal | 318 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | ground | 10 | vertical | 318 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | ground | 20 | horizontal | 288 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | ground | 20 | vertical | 288 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-03 | ground | 30 | horizontal | 262 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | 2026-10-03 | ground | 30 | vertical | 262 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | airborne | 10 | horizontal | 476 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | airborne | 10 | vertical | 476 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | airborne | 20 | horizontal | 396 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | airborne | 20 | vertical | 396 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | airborne | 30 | horizontal | 318 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | 2026-10-04 | airborne | 30 | vertical | 318 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | all | 10 | horizontal | 620 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | all | 10 | vertical | 620 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | all | 20 | horizontal | 514 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | all | 20 | vertical | 514 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | all | 30 | horizontal | 422 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | 2026-10-04 | all | 30 | vertical | 422 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | ground | 10 | horizontal | 144 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | ground | 10 | vertical | 144 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | ground | 20 | horizontal | 118 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | ground | 20 | vertical | 118 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | 2026-10-04 | ground | 30 | horizontal | 104 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | 2026-10-04 | ground | 30 | vertical | 104 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | airborne | 10 | horizontal | 1078 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | airborne | 10 | vertical | 1078 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | airborne | 20 | horizontal | 915 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | airborne | 20 | vertical | 915 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | airborne | 30 | horizontal | 724 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | combined | airborne | 30 | vertical | 724 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | all | 10 | horizontal | 1540 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | all | 10 | vertical | 1540 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | all | 20 | horizontal | 1321 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | all | 20 | vertical | 1321 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | all | 30 | horizontal | 1090 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | combined | all | 30 | vertical | 1090 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | ground | 10 | horizontal | 462 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | ground | 10 | vertical | 462 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | ground | 20 | horizontal | 406 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | ground | 20 | vertical | 406 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| gt_replay | combined | ground | 30 | horizontal | 366 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gt_replay | combined | ground | 30 | vertical | 366 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
