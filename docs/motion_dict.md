# Motion dictionary validation

This is GT-conditioned token reconstruction error, not prediction error.

Selected shift=6, K=2048, heading=chord.
30 s autoregressive horizontal FDE: 2.5599 m. Target <10 m: met.

Provisional advisor decisions: tolerance starts at 0.5 internal units and retries ×0.8; token zero exactly stationary; training segments filtered at 200 m/s horizontal / 40 m/s vertical; seeded shared pool capped at 500k per shift. See ../SYGARD_QA_LOG.md.

Chord heading is computed from the preceding segment only (causal for history/input), with data fallback on ground or speed ≤10 m/s. All token subframes are retained at 3 Hz. An incomplete final horizon token is matched at the exact requested subframe, with no GT beyond the horizon.

| shift | heading | K | mode | horizon s | group | agents | hADE m | hFDE m | vADE m | vFDE m |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | chord | 1024 | autoregressive | 10 | all | 920 | 1.3776 | 1.5268 | 0.6695 | 0.5412 |
| 3 | chord | 1024 | autoregressive | 10 | ground | 318 | 1.1679 | 1.5490 | 0.2795 | 0.2419 |
| 3 | chord | 1024 | autoregressive | 10 | airborne | 602 | 1.4884 | 1.5151 | 0.8756 | 0.6994 |
| 3 | chord | 1024 | autoregressive | 30 | all | 668 | 1.5584 | 1.8880 | 0.7210 | 0.7686 |
| 3 | chord | 1024 | autoregressive | 30 | ground | 262 | 1.5135 | 1.7289 | 0.4177 | 0.4741 |
| 3 | chord | 1024 | autoregressive | 30 | airborne | 406 | 1.5873 | 1.9906 | 0.9167 | 0.9586 |
| 3 | chord | 1024 | one_shot | 10 | all | 920 | 0.9848 | 1.1219 | 0.5048 | 0.4136 |
| 3 | chord | 1024 | one_shot | 10 | ground | 318 | 0.7435 | 1.0194 | 0.1606 | 0.1435 |
| 3 | chord | 1024 | one_shot | 10 | airborne | 602 | 1.1122 | 1.1761 | 0.6867 | 0.5563 |
| 3 | chord | 1024 | one_shot | 30 | all | 668 | 1.0135 | 1.1561 | 0.5255 | 0.5733 |
| 3 | chord | 1024 | one_shot | 30 | ground | 262 | 0.8026 | 1.0309 | 0.2568 | 0.3540 |
| 3 | chord | 1024 | one_shot | 30 | airborne | 406 | 1.1496 | 1.2369 | 0.6988 | 0.7149 |
| 3 | data | 1024 | autoregressive | 10 | all | 920 | 1.4842 | 1.6471 | 0.7294 | 0.6245 |
| 3 | data | 1024 | autoregressive | 10 | ground | 318 | 1.2361 | 1.6715 | 0.2929 | 0.2810 |
| 3 | data | 1024 | autoregressive | 10 | airborne | 602 | 1.6152 | 1.6342 | 0.9600 | 0.8060 |
| 3 | data | 1024 | autoregressive | 30 | all | 668 | 1.7577 | 2.1194 | 0.7979 | 1.0707 |
| 3 | data | 1024 | autoregressive | 30 | ground | 262 | 1.6619 | 2.0265 | 0.4403 | 0.4754 |
| 3 | data | 1024 | autoregressive | 30 | airborne | 406 | 1.8195 | 2.1793 | 1.0286 | 1.4548 |
| 3 | data | 1024 | one_shot | 10 | all | 920 | 0.9356 | 1.1143 | 0.5328 | 0.4506 |
| 3 | data | 1024 | one_shot | 10 | ground | 318 | 0.7210 | 0.9820 | 0.1712 | 0.1551 |
| 3 | data | 1024 | one_shot | 10 | airborne | 602 | 1.0489 | 1.1842 | 0.7238 | 0.6067 |
| 3 | data | 1024 | one_shot | 30 | all | 668 | 0.9591 | 1.1193 | 0.5432 | 0.5764 |
| 3 | data | 1024 | one_shot | 30 | ground | 262 | 0.7849 | 1.0103 | 0.2552 | 0.3363 |
| 3 | data | 1024 | one_shot | 30 | airborne | 406 | 1.0715 | 1.1897 | 0.7290 | 0.7314 |
| 6 | chord | 1024 | autoregressive | 10 | all | 920 | 2.5804 | 2.8672 | 1.4071 | 1.2549 |
| 6 | chord | 1024 | autoregressive | 10 | ground | 318 | 2.0359 | 2.6573 | 0.4558 | 0.4390 |
| 6 | chord | 1024 | autoregressive | 10 | airborne | 602 | 2.8681 | 2.9781 | 1.9096 | 1.6858 |
| 6 | chord | 1024 | autoregressive | 30 | all | 668 | 3.2144 | 3.6675 | 1.6292 | 2.1489 |
| 6 | chord | 1024 | autoregressive | 30 | ground | 262 | 3.0266 | 3.8902 | 0.8234 | 1.1289 |
| 6 | chord | 1024 | autoregressive | 30 | airborne | 406 | 3.3355 | 3.5237 | 2.1492 | 2.8072 |
| 6 | chord | 1024 | one_shot | 10 | all | 920 | 1.9083 | 2.0544 | 1.0498 | 0.7896 |
| 6 | chord | 1024 | one_shot | 10 | ground | 318 | 1.3457 | 1.8656 | 0.3054 | 0.2392 |
| 6 | chord | 1024 | one_shot | 10 | airborne | 602 | 2.2055 | 2.1541 | 1.4430 | 1.0804 |
| 6 | chord | 1024 | one_shot | 30 | all | 668 | 1.9588 | 2.2231 | 1.1078 | 1.2128 |
| 6 | chord | 1024 | one_shot | 30 | ground | 262 | 1.5035 | 2.0102 | 0.5250 | 0.5839 |
| 6 | chord | 1024 | one_shot | 30 | airborne | 406 | 2.2526 | 2.3604 | 1.4839 | 1.6187 |
| 6 | data | 1024 | autoregressive | 10 | all | 920 | 2.6744 | 3.1033 | 1.4860 | 1.3153 |
| 6 | data | 1024 | autoregressive | 10 | ground | 318 | 2.0577 | 2.8738 | 0.4716 | 0.3938 |
| 6 | data | 1024 | autoregressive | 10 | airborne | 602 | 3.0002 | 3.2245 | 2.0218 | 1.8022 |
| 6 | data | 1024 | autoregressive | 30 | all | 668 | 3.7050 | 5.4463 | 1.6203 | 1.8952 |
| 6 | data | 1024 | autoregressive | 30 | ground | 262 | 3.0804 | 4.0229 | 0.8114 | 0.9937 |
| 6 | data | 1024 | autoregressive | 30 | airborne | 406 | 4.1081 | 6.3649 | 2.1423 | 2.4769 |
| 6 | data | 1024 | one_shot | 10 | all | 920 | 1.8308 | 1.9556 | 1.0583 | 0.8271 |
| 6 | data | 1024 | one_shot | 10 | ground | 318 | 1.2398 | 1.7242 | 0.3231 | 0.2569 |
| 6 | data | 1024 | one_shot | 10 | airborne | 602 | 2.1430 | 2.0778 | 1.4467 | 1.1283 |
| 6 | data | 1024 | one_shot | 30 | all | 668 | 1.8699 | 2.0465 | 1.0932 | 1.2419 |
| 6 | data | 1024 | one_shot | 30 | ground | 262 | 1.3844 | 1.7823 | 0.5381 | 0.6083 |
| 6 | data | 1024 | one_shot | 30 | airborne | 406 | 2.1832 | 2.2170 | 1.4515 | 1.6508 |
| 9 | chord | 1024 | autoregressive | 10 | all | 920 | 3.3303 | 2.4197 | 1.9487 | 1.1825 |
| 9 | chord | 1024 | autoregressive | 10 | ground | 318 | 2.2603 | 2.0022 | 0.6313 | 0.3062 |
| 9 | chord | 1024 | autoregressive | 10 | airborne | 602 | 3.8955 | 2.6402 | 2.6446 | 1.6454 |
| 9 | chord | 1024 | autoregressive | 30 | all | 668 | 5.1313 | 7.5185 | 2.3560 | 2.7674 |
| 9 | chord | 1024 | autoregressive | 30 | ground | 262 | 4.3204 | 7.3320 | 1.1102 | 1.4602 |
| 9 | chord | 1024 | autoregressive | 30 | airborne | 406 | 5.6547 | 7.6389 | 3.1600 | 3.6110 |
| 9 | chord | 1024 | one_shot | 10 | all | 920 | 2.4302 | 1.0143 | 1.4586 | 0.4579 |
| 9 | chord | 1024 | one_shot | 10 | ground | 318 | 1.5264 | 0.5724 | 0.4759 | 0.1425 |
| 9 | chord | 1024 | one_shot | 10 | airborne | 602 | 2.9077 | 1.2477 | 1.9777 | 0.6246 |
| 9 | chord | 1024 | one_shot | 30 | all | 668 | 2.6525 | 3.1396 | 1.6062 | 1.9916 |
| 9 | chord | 1024 | one_shot | 30 | ground | 262 | 1.8622 | 2.1128 | 0.7873 | 0.9084 |
| 9 | chord | 1024 | one_shot | 30 | airborne | 406 | 3.1625 | 3.8022 | 2.1347 | 2.6907 |
| 9 | data | 1024 | autoregressive | 10 | all | 920 | 3.4268 | 3.0294 | 1.8663 | 1.0121 |
| 9 | data | 1024 | autoregressive | 10 | ground | 318 | 2.2048 | 2.0730 | 0.6343 | 0.2953 |
| 9 | data | 1024 | autoregressive | 10 | airborne | 602 | 4.0724 | 3.5347 | 2.5171 | 1.3908 |
| 9 | data | 1024 | autoregressive | 30 | all | 668 | 5.3735 | 8.6124 | 2.3138 | 2.9514 |
| 9 | data | 1024 | autoregressive | 30 | ground | 262 | 4.2716 | 6.0555 | 1.1637 | 1.7366 |
| 9 | data | 1024 | autoregressive | 30 | airborne | 406 | 6.0846 | 10.2624 | 3.0560 | 3.7354 |
| 9 | data | 1024 | one_shot | 10 | all | 920 | 2.3640 | 0.8823 | 1.4273 | 0.4212 |
| 9 | data | 1024 | one_shot | 10 | ground | 318 | 1.4982 | 0.5847 | 0.4431 | 0.1321 |
| 9 | data | 1024 | one_shot | 10 | airborne | 602 | 2.8214 | 1.0395 | 1.9472 | 0.5739 |
| 9 | data | 1024 | one_shot | 30 | all | 668 | 2.6281 | 3.1140 | 1.5402 | 1.8650 |
| 9 | data | 1024 | one_shot | 30 | ground | 262 | 1.8684 | 2.3347 | 0.7427 | 0.8921 |
| 9 | data | 1024 | one_shot | 30 | airborne | 406 | 3.1184 | 3.6168 | 2.0549 | 2.4928 |
| 3 | chord | 2048 | autoregressive | 10 | all | 920 | 1.2736 | 1.3054 | 0.6508 | 0.4780 |
| 3 | chord | 2048 | autoregressive | 10 | ground | 318 | 1.0881 | 1.4148 | 0.2561 | 0.2323 |
| 3 | chord | 2048 | autoregressive | 10 | airborne | 602 | 1.3716 | 1.2476 | 0.8593 | 0.6077 |
| 3 | chord | 2048 | autoregressive | 30 | all | 668 | 1.4001 | 1.4945 | 0.6907 | 0.6417 |
| 3 | chord | 2048 | autoregressive | 30 | ground | 262 | 1.3553 | 1.5675 | 0.3798 | 0.3646 |
| 3 | chord | 2048 | autoregressive | 30 | airborne | 406 | 1.4291 | 1.4473 | 0.8913 | 0.8206 |
| 3 | chord | 2048 | one_shot | 10 | all | 920 | 0.9655 | 1.0620 | 0.4940 | 0.3894 |
| 3 | chord | 2048 | one_shot | 10 | ground | 318 | 0.7455 | 1.0070 | 0.1542 | 0.1362 |
| 3 | chord | 2048 | one_shot | 10 | airborne | 602 | 1.0817 | 1.0911 | 0.6735 | 0.5232 |
| 3 | chord | 2048 | one_shot | 30 | all | 668 | 0.9937 | 1.0837 | 0.5157 | 0.5206 |
| 3 | chord | 2048 | one_shot | 30 | ground | 262 | 0.8033 | 1.0120 | 0.2478 | 0.3429 |
| 3 | chord | 2048 | one_shot | 30 | airborne | 406 | 1.1166 | 1.1300 | 0.6886 | 0.6353 |
| 3 | data | 2048 | autoregressive | 10 | all | 920 | 1.3517 | 1.4081 | 0.6379 | 0.4496 |
| 3 | data | 2048 | autoregressive | 10 | ground | 318 | 1.1347 | 1.4895 | 0.2555 | 0.2342 |
| 3 | data | 2048 | autoregressive | 10 | airborne | 602 | 1.4664 | 1.3651 | 0.8400 | 0.5634 |
| 3 | data | 2048 | autoregressive | 30 | all | 668 | 1.5671 | 1.6447 | 0.6841 | 0.6554 |
| 3 | data | 2048 | autoregressive | 30 | ground | 262 | 1.5250 | 1.7365 | 0.3872 | 0.3696 |
| 3 | data | 2048 | autoregressive | 30 | airborne | 406 | 1.5943 | 1.5855 | 0.8757 | 0.8399 |
| 3 | data | 2048 | one_shot | 10 | all | 920 | 0.9085 | 1.0265 | 0.5087 | 0.4241 |
| 3 | data | 2048 | one_shot | 10 | ground | 318 | 0.6954 | 0.9281 | 0.1635 | 0.1577 |
| 3 | data | 2048 | one_shot | 10 | airborne | 602 | 1.0211 | 1.0785 | 0.6910 | 0.5648 |
| 3 | data | 2048 | one_shot | 30 | all | 668 | 0.9277 | 1.0641 | 0.5201 | 0.4799 |
| 3 | data | 2048 | one_shot | 30 | ground | 262 | 0.7574 | 0.9486 | 0.2473 | 0.3199 |
| 3 | data | 2048 | one_shot | 30 | airborne | 406 | 1.0375 | 1.1386 | 0.6961 | 0.5832 |
| 6 | chord | 2048 | autoregressive | 10 | all | 920 | 2.3250 | 2.2430 | 1.2221 | 0.8880 |
| 6 | chord | 2048 | autoregressive | 10 | ground | 318 | 1.8693 | 2.3139 | 0.3877 | 0.3481 |
| 6 | chord | 2048 | autoregressive | 10 | airborne | 602 | 2.5658 | 2.2056 | 1.6628 | 1.1732 |
| 6 | chord | 2048 | autoregressive | 30 | all | 668 | 2.6353 | 2.5599 | 1.3439 | 1.3112 |
| 6 | chord | 2048 | autoregressive | 30 | ground | 262 | 2.4646 | 2.7935 | 0.6983 | 0.6871 |
| 6 | chord | 2048 | autoregressive | 30 | airborne | 406 | 2.7455 | 2.4091 | 1.7605 | 1.7139 |
| 6 | chord | 2048 | one_shot | 10 | all | 920 | 1.8500 | 1.8726 | 1.0154 | 0.7184 |
| 6 | chord | 2048 | one_shot | 10 | ground | 318 | 1.2960 | 1.7765 | 0.2906 | 0.2536 |
| 6 | chord | 2048 | one_shot | 10 | airborne | 602 | 2.1427 | 1.9233 | 1.3983 | 0.9639 |
| 6 | chord | 2048 | one_shot | 30 | all | 668 | 1.8988 | 1.9175 | 1.0611 | 1.0094 |
| 6 | chord | 2048 | one_shot | 30 | ground | 262 | 1.4597 | 1.7923 | 0.4926 | 0.4975 |
| 6 | chord | 2048 | one_shot | 30 | airborne | 406 | 2.1821 | 1.9983 | 1.4279 | 1.3397 |
| 6 | data | 2048 | autoregressive | 10 | all | 920 | 2.4093 | 2.3813 | 1.2202 | 0.8903 |
| 6 | data | 2048 | autoregressive | 10 | ground | 318 | 1.9185 | 2.3774 | 0.4034 | 0.3770 |
| 6 | data | 2048 | autoregressive | 10 | airborne | 602 | 2.6685 | 2.3833 | 1.6516 | 1.1615 |
| 6 | data | 2048 | autoregressive | 30 | all | 668 | 2.8114 | 2.8696 | 1.3594 | 1.3292 |
| 6 | data | 2048 | autoregressive | 30 | ground | 262 | 2.5845 | 2.7142 | 0.6874 | 0.7069 |
| 6 | data | 2048 | autoregressive | 30 | airborne | 406 | 2.9578 | 2.9700 | 1.7930 | 1.7307 |
| 6 | data | 2048 | one_shot | 10 | all | 920 | 1.8055 | 1.8217 | 0.9918 | 0.6929 |
| 6 | data | 2048 | one_shot | 10 | ground | 318 | 1.2422 | 1.6603 | 0.2874 | 0.2522 |
| 6 | data | 2048 | one_shot | 10 | airborne | 602 | 2.1031 | 1.9069 | 1.3639 | 0.9257 |
| 6 | data | 2048 | one_shot | 30 | all | 668 | 1.8588 | 1.9254 | 1.0464 | 0.9253 |
| 6 | data | 2048 | one_shot | 30 | ground | 262 | 1.4021 | 1.7485 | 0.4867 | 0.4737 |
| 6 | data | 2048 | one_shot | 30 | airborne | 406 | 2.1535 | 2.0395 | 1.4075 | 1.2168 |
| 9 | chord | 2048 | autoregressive | 10 | all | 920 | 2.9659 | 1.7229 | 1.6468 | 0.6643 |
| 9 | chord | 2048 | autoregressive | 10 | ground | 318 | 2.1507 | 1.6904 | 0.5020 | 0.2591 |
| 9 | chord | 2048 | autoregressive | 10 | airborne | 602 | 3.3965 | 1.7401 | 2.2516 | 0.8784 |
| 9 | chord | 2048 | autoregressive | 30 | all | 668 | 3.9807 | 4.8432 | 1.9265 | 1.7461 |
| 9 | chord | 2048 | autoregressive | 30 | ground | 262 | 3.6611 | 5.3018 | 0.9498 | 0.9363 |
| 9 | chord | 2048 | autoregressive | 30 | airborne | 406 | 4.1869 | 4.5472 | 2.5568 | 2.2686 |
| 9 | chord | 2048 | one_shot | 10 | all | 920 | 2.3151 | 0.8202 | 1.3567 | 0.3622 |
| 9 | chord | 2048 | one_shot | 10 | ground | 318 | 1.4863 | 0.5172 | 0.4128 | 0.1123 |
| 9 | chord | 2048 | one_shot | 10 | airborne | 602 | 2.7529 | 0.9802 | 1.8553 | 0.4942 |
| 9 | chord | 2048 | one_shot | 30 | all | 668 | 2.5528 | 2.5142 | 1.4860 | 1.3940 |
| 9 | chord | 2048 | one_shot | 30 | ground | 262 | 1.8607 | 1.8740 | 0.7246 | 0.6808 |
| 9 | chord | 2048 | one_shot | 30 | airborne | 406 | 2.9994 | 2.9274 | 1.9773 | 1.8542 |
| 9 | data | 2048 | autoregressive | 10 | all | 920 | 2.9143 | 1.7406 | 1.6263 | 0.7162 |
| 9 | data | 2048 | autoregressive | 10 | ground | 318 | 2.0630 | 1.6109 | 0.5130 | 0.2344 |
| 9 | data | 2048 | autoregressive | 10 | airborne | 602 | 3.3641 | 1.8091 | 2.2144 | 0.9707 |
| 9 | data | 2048 | autoregressive | 30 | all | 668 | 3.7748 | 4.4672 | 2.0087 | 2.1869 |
| 9 | data | 2048 | autoregressive | 30 | ground | 262 | 3.2900 | 4.2775 | 1.0111 | 1.1686 |
| 9 | data | 2048 | autoregressive | 30 | airborne | 406 | 4.0876 | 4.5895 | 2.6524 | 2.8440 |
| 9 | data | 2048 | one_shot | 10 | all | 920 | 2.2609 | 0.7442 | 1.3430 | 0.3089 |
| 9 | data | 2048 | one_shot | 10 | ground | 318 | 1.4450 | 0.5210 | 0.4262 | 0.0869 |
| 9 | data | 2048 | one_shot | 10 | airborne | 602 | 2.6919 | 0.8621 | 1.8273 | 0.4261 |
| 9 | data | 2048 | one_shot | 30 | all | 668 | 2.4956 | 2.5544 | 1.4836 | 1.5196 |
| 9 | data | 2048 | one_shot | 30 | ground | 262 | 1.8207 | 2.0335 | 0.7170 | 0.6621 |
| 9 | data | 2048 | one_shot | 30 | airborne | 406 | 2.9311 | 2.8906 | 1.9784 | 2.0729 |
| 3 | chord | 512 | autoregressive | 10 | all | 920 | 1.6289 | 2.0450 | 0.8103 | 0.6988 |
| 3 | chord | 512 | autoregressive | 10 | ground | 318 | 1.3622 | 2.1242 | 0.3619 | 0.3522 |
| 3 | chord | 512 | autoregressive | 10 | airborne | 602 | 1.7698 | 2.0031 | 1.0472 | 0.8818 |
| 3 | chord | 512 | autoregressive | 30 | all | 668 | 2.2097 | 3.4747 | 0.8549 | 1.0718 |
| 3 | chord | 512 | autoregressive | 30 | ground | 262 | 2.1386 | 3.0104 | 0.5051 | 0.5776 |
| 3 | chord | 512 | autoregressive | 30 | airborne | 406 | 2.2556 | 3.7744 | 1.0806 | 1.3908 |
| 3 | chord | 512 | one_shot | 10 | all | 920 | 1.0484 | 1.2319 | 0.5564 | 0.5197 |
| 3 | chord | 512 | one_shot | 10 | ground | 318 | 0.7676 | 1.0559 | 0.1810 | 0.1555 |
| 3 | chord | 512 | one_shot | 10 | airborne | 602 | 1.1967 | 1.3248 | 0.7547 | 0.7120 |
| 3 | chord | 512 | one_shot | 30 | all | 668 | 1.0793 | 1.3401 | 0.5732 | 0.6857 |
| 3 | chord | 512 | one_shot | 30 | ground | 262 | 0.8393 | 1.1172 | 0.2834 | 0.4559 |
| 3 | chord | 512 | one_shot | 30 | airborne | 406 | 1.2342 | 1.4840 | 0.7602 | 0.8340 |
| 3 | data | 512 | autoregressive | 10 | all | 920 | 1.7783 | 2.2130 | 0.8415 | 0.7177 |
| 3 | data | 512 | autoregressive | 10 | ground | 318 | 1.4851 | 2.2243 | 0.3569 | 0.2738 |
| 3 | data | 512 | autoregressive | 10 | airborne | 602 | 1.9332 | 2.2070 | 1.0974 | 0.9522 |
| 3 | data | 512 | autoregressive | 30 | all | 668 | 2.5740 | 3.7825 | 0.9086 | 1.2086 |
| 3 | data | 512 | autoregressive | 30 | ground | 262 | 2.3637 | 3.3625 | 0.5119 | 0.5703 |
| 3 | data | 512 | autoregressive | 30 | airborne | 406 | 2.7097 | 4.0535 | 1.1645 | 1.6205 |
| 3 | data | 512 | one_shot | 10 | all | 920 | 0.9832 | 1.2135 | 0.5759 | 0.5258 |
| 3 | data | 512 | one_shot | 10 | ground | 318 | 0.7380 | 1.0002 | 0.1915 | 0.1813 |
| 3 | data | 512 | one_shot | 10 | airborne | 602 | 1.1127 | 1.3262 | 0.7790 | 0.7078 |
| 3 | data | 512 | one_shot | 30 | all | 668 | 1.0110 | 1.2984 | 0.5867 | 0.6562 |
| 3 | data | 512 | one_shot | 30 | ground | 262 | 0.8020 | 1.0886 | 0.2886 | 0.4051 |
| 3 | data | 512 | one_shot | 30 | airborne | 406 | 1.1458 | 1.4337 | 0.7791 | 0.8183 |
| 6 | chord | 512 | autoregressive | 10 | all | 920 | 3.1708 | 4.2318 | 1.5651 | 1.5337 |
| 6 | chord | 512 | autoregressive | 10 | ground | 318 | 2.2672 | 3.2682 | 0.5421 | 0.5476 |
| 6 | chord | 512 | autoregressive | 10 | airborne | 602 | 3.6481 | 4.7408 | 2.1056 | 2.0546 |
| 6 | chord | 512 | autoregressive | 30 | all | 668 | 4.9458 | 8.5083 | 1.8423 | 2.3462 |
| 6 | chord | 512 | autoregressive | 30 | ground | 262 | 3.8333 | 5.4522 | 0.8939 | 1.3228 |
| 6 | chord | 512 | autoregressive | 30 | airborne | 406 | 5.6638 | 10.4804 | 2.4544 | 3.0067 |
| 6 | chord | 512 | one_shot | 10 | all | 920 | 2.0167 | 2.3885 | 1.1431 | 1.0103 |
| 6 | chord | 512 | one_shot | 10 | ground | 318 | 1.3534 | 1.9242 | 0.3598 | 0.2638 |
| 6 | chord | 512 | one_shot | 10 | airborne | 602 | 2.3671 | 2.6338 | 1.5569 | 1.4046 |
| 6 | chord | 512 | one_shot | 30 | all | 668 | 2.1048 | 2.8252 | 1.1868 | 1.5622 |
| 6 | chord | 512 | one_shot | 30 | ground | 262 | 1.5601 | 2.1557 | 0.5622 | 0.7470 |
| 6 | chord | 512 | one_shot | 30 | airborne | 406 | 2.4563 | 3.2572 | 1.5899 | 2.0883 |
| 6 | data | 512 | autoregressive | 10 | all | 920 | 3.2826 | 4.5026 | 1.6258 | 1.6339 |
| 6 | data | 512 | autoregressive | 10 | ground | 318 | 2.1678 | 3.1705 | 0.4909 | 0.4674 |
| 6 | data | 512 | autoregressive | 10 | airborne | 602 | 3.8715 | 5.2063 | 2.2253 | 2.2500 |
| 6 | data | 512 | autoregressive | 30 | all | 668 | 5.6330 | 9.6074 | 1.9365 | 2.4660 |
| 6 | data | 512 | autoregressive | 30 | ground | 262 | 4.3321 | 6.9492 | 0.9358 | 1.1737 |
| 6 | data | 512 | autoregressive | 30 | airborne | 406 | 6.4725 | 11.3228 | 2.5822 | 3.2999 |
| 6 | data | 512 | one_shot | 10 | all | 920 | 1.9562 | 2.3432 | 1.1285 | 1.0183 |
| 6 | data | 512 | one_shot | 10 | ground | 318 | 1.2821 | 1.7863 | 0.3582 | 0.3409 |
| 6 | data | 512 | one_shot | 10 | airborne | 602 | 2.3123 | 2.6374 | 1.5354 | 1.3762 |
| 6 | data | 512 | one_shot | 30 | all | 668 | 2.0162 | 2.5262 | 1.1487 | 1.5094 |
| 6 | data | 512 | one_shot | 30 | ground | 262 | 1.4745 | 1.9620 | 0.5722 | 0.9003 |
| 6 | data | 512 | one_shot | 30 | airborne | 406 | 2.3658 | 2.8903 | 1.5207 | 1.9024 |
| 9 | chord | 512 | autoregressive | 10 | all | 920 | 3.9947 | 3.8608 | 2.2790 | 1.8846 |
| 9 | chord | 512 | autoregressive | 10 | ground | 318 | 2.5173 | 2.7231 | 0.8444 | 0.5315 |
| 9 | chord | 512 | autoregressive | 10 | airborne | 602 | 4.7751 | 4.4618 | 3.0368 | 2.5994 |
| 9 | chord | 512 | autoregressive | 30 | all | 668 | 6.9293 | 11.8017 | 3.0521 | 4.7620 |
| 9 | chord | 512 | autoregressive | 30 | ground | 262 | 5.1044 | 9.0800 | 1.4876 | 2.1318 |
| 9 | chord | 512 | autoregressive | 30 | airborne | 406 | 8.1070 | 13.5581 | 4.0617 | 6.4594 |
| 9 | chord | 512 | one_shot | 10 | all | 920 | 2.6090 | 1.2436 | 1.5754 | 0.6336 |
| 9 | chord | 512 | one_shot | 10 | ground | 318 | 1.5366 | 0.6107 | 0.5489 | 0.1863 |
| 9 | chord | 512 | one_shot | 10 | airborne | 602 | 3.1755 | 1.5779 | 2.1176 | 0.8699 |
| 9 | chord | 512 | one_shot | 30 | all | 668 | 2.8452 | 3.8031 | 1.7528 | 2.7781 |
| 9 | chord | 512 | one_shot | 30 | ground | 262 | 1.9027 | 2.3010 | 0.9070 | 1.3166 |
| 9 | chord | 512 | one_shot | 30 | airborne | 406 | 3.4534 | 4.7725 | 2.2987 | 3.7213 |
| 9 | data | 512 | autoregressive | 10 | all | 920 | 4.0894 | 4.4352 | 2.4411 | 2.0678 |
| 9 | data | 512 | autoregressive | 10 | ground | 318 | 2.4701 | 2.6638 | 0.7240 | 0.5042 |
| 9 | data | 512 | autoregressive | 10 | airborne | 602 | 4.9448 | 5.3709 | 3.3482 | 2.8937 |
| 9 | data | 512 | autoregressive | 30 | all | 668 | 6.6736 | 11.0242 | 3.0566 | 4.3656 |
| 9 | data | 512 | autoregressive | 30 | ground | 262 | 5.0517 | 8.3951 | 1.4049 | 2.2600 |
| 9 | data | 512 | autoregressive | 30 | airborne | 406 | 7.7202 | 12.7208 | 4.1225 | 5.7243 |
| 9 | data | 512 | one_shot | 10 | all | 920 | 2.5439 | 1.0949 | 1.5684 | 0.5389 |
| 9 | data | 512 | one_shot | 10 | ground | 318 | 1.5517 | 0.7160 | 0.4940 | 0.1152 |
| 9 | data | 512 | one_shot | 10 | airborne | 602 | 3.0679 | 1.2950 | 2.1359 | 0.7628 |
| 9 | data | 512 | one_shot | 30 | all | 668 | 2.7875 | 3.7994 | 1.7089 | 2.5329 |
| 9 | data | 512 | one_shot | 30 | ground | 262 | 1.9408 | 2.6708 | 0.8336 | 1.1913 |
| 9 | data | 512 | one_shot | 30 | airborne | 406 | 3.3340 | 4.5277 | 2.2737 | 3.3986 |

## Selected dictionary metadata

```json
{
  "shift": 6,
  "pos_scale": 10.0,
  "radius": 2.0,
  "lambda_z": 1.0,
  "K": 2048,
  "git_hash": "1b7b6323aa00ffe13228b8ab549c110300127a13",
  "heading_source": "chord",
  "tol": 0.5,
  "seed": 42,
  "max_vertical_speed": 40.0,
  "max_horizontal_speed": 200.0,
  "unclustered_fraction": 0.007942122377643163,
  "candidate_segments": 630450,
  "invalid_segments": 462006,
  "vertical_outliers": 221,
  "horizontal_outliers": 6,
  "n_segments_total": 168217,
  "n_segments_used": 168217
}
```

Full sweep and per-dictionary filtering metadata: data/motion_dict/eval/sweep.json (ignored generated output).

## Default training chain

Default training chains from frame0, with no current reset. Forecast rollouts start at tokenized current; forecast metrics compare raw GT. M3 parity tests deliberately start from raw current frame18, matching the M3 evaluator.

Current tokenized pose drift: {"n_agents": 920, "current_position_3d_drift_mean_m": 2.556192735145079, "current_position_3d_drift_p95_m": 4.135660019284286}

Raw-GT default-chain reconstruction (report only):

```json
[
  {
    "horizon_s": 10,
    "group": "all",
    "n_agents": 920,
    "horizontal_ade_m": 2.6219914075302397,
    "horizontal_fde_m": 2.4214893468747,
    "vertical_ade_m": 1.3233334763229647,
    "vertical_fde_m": 1.0067223461678505
  },
  {
    "horizon_s": 10,
    "group": "ground",
    "n_agents": 318,
    "horizontal_ade_m": 2.395957590778319,
    "horizontal_fde_m": 2.556270915162453,
    "vertical_ade_m": 0.4527494069874536,
    "vertical_fde_m": 0.4551256923486388
  },
  {
    "horizon_s": 10,
    "group": "airborne",
    "n_agents": 602,
    "horizontal_ade_m": 2.7413913306649755,
    "horizontal_fde_m": 2.3502924387094093,
    "vertical_ade_m": 1.7832101109553444,
    "vertical_fde_m": 1.2980973227700252
  },
  {
    "horizon_s": 30,
    "group": "all",
    "n_agents": 668,
    "horizontal_ade_m": 2.768703842759704,
    "horizontal_fde_m": 2.591672749458043,
    "vertical_ade_m": 1.3815668671840928,
    "vertical_fde_m": 1.3016280259285893
  },
  {
    "horizon_s": 30,
    "group": "ground",
    "n_agents": 262,
    "horizontal_ade_m": 2.721580793800143,
    "horizontal_fde_m": 2.902660208620732,
    "vertical_ade_m": 0.7309075530677838,
    "vertical_fde_m": 0.6746529733303316
  },
  {
    "horizon_s": 30,
    "group": "airborne",
    "n_agents": 406,
    "horizontal_ade_m": 2.7991132979996185,
    "horizontal_fde_m": 2.390986261032859,
    "vertical_ade_m": 1.8014504639783608,
    "vertical_fde_m": 1.7062276904131792
  }
]
```
