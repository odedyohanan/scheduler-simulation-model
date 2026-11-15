# One-step evaluation — `models/mlp_h4.pt`

Mean and median absolute error in real units (ms, batches/s, core-ms per window, °C). Lower is better. A mean far above the median means a few huge misses dominate.

*Noise floor: the error left even when the simulator's full hidden state is known (queued requests, burst schedule), measured on 50 adaptive test states by re-simulating each window with 20 seeds. A model that sees only telemetry can't reach it; it is a lower bound, not a target.

## adaptive (3570 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 5.82 | 6.36 | 8.67 | 0.574 | 0.674 | 0.781 | 0.913 |
| a.queue_work_ms | 66.5 | 31.7 | 21.8 | 1.21 | 1.65 | 1.22 | 0.279 |
| a.cpu_cores | 0.198 | 0.207 | — | 0.126 | 0.11 | — | 0.941 |
| b.rate | 23 | 96.8 | 95 | 12.9 | 5.98 | 95.6 | 0.954 |
| b.buffer | 1.83 | 2.97 | — | 0.872 | 0.379 | — | 0.731 |
| c.work_core_ms | 51.8 | 349 | 31.9 | 34 | 41.2 | 16.3 | 0.987 |
| gpu_temp_0 | 0.0462 | 0.309 | 0.213 | 0.032 | 0.15 | 0.14 | 1.000 |
| gpu_temp_1 | 0.0868 | 0.415 | 0.148 | 0.0504 | 0.33 | 0.13 | 1.000 |

Overloaded states only (A queue > 100 ms): 174 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 88.9 | 70.7 | 172 | 36 | 35 | 84.4 | 0.852 |
| a.queue_work_ms | 1.25e+03 | 519 | 330 | 361 | 317 | 219 | -0.150 |
| a.cpu_cores | 0.28 | 0.416 | — | 0.204 | 0.0945 | — | 0.790 |
| b.rate | 45.4 | 90.3 | 197 | 34.9 | 37.5 | 193 | 0.759 |
| b.buffer | 2.06 | 2.72 | — | 0.541 | 0 | — | 0.425 |
| c.work_core_ms | 67.8 | 259 | 31 | 55.1 | 15.2 | 25 | 0.967 |
| gpu_temp_0 | 0.0927 | 0.286 | 0.867 | 0.0721 | 0.2 | 0.98 | 1.000 |
| gpu_temp_1 | 0.162 | 0.696 | 0.256 | 0.108 | 0.73 | 0.2 | 0.999 |

## mpc (5950 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 14.9 | 5.19 | 6.53 | 3.21 | 0.518 | 0.911 | 0.471 |
| a.queue_work_ms | 402 | 24.2 | 20.5 | 3.19 | 1.74 | 1.35 | -178.572 |
| a.cpu_cores | 0.542 | 0.2 | — | 0.411 | 0.11 | — | 0.600 |
| b.rate | 103 | 58.7 | 61.5 | 78.1 | 6.85 | 61.3 | 0.690 |
| b.buffer | 5.63 | 2.81 | — | 4.64 | 0.357 | — | -0.094 |
| c.work_core_ms | 304 | 251 | 90 | 219 | 46.4 | 65.2 | 0.724 |
| gpu_temp_0 | 0.222 | 0.326 | 0.176 | 0.18 | 0.22 | 0.14 | 0.999 |
| gpu_temp_1 | 0.382 | 0.381 | 0.148 | 0.305 | 0.31 | 0.12 | 0.997 |

Overloaded states only (A queue > 100 ms): 242 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 210 | 66.4 | 163 | 66.8 | 31.5 | 91.2 | 0.136 |
| a.queue_work_ms | 9.49e+03 | 430 | 354 | 448 | 294 | 299 | -270.545 |
| a.cpu_cores | 0.964 | 0.465 | — | 0.856 | 0.0838 | — | -1.690 |
| b.rate | 114 | 132 | 25.7 | 43.5 | 0.876 | 6.35 | 0.645 |
| b.buffer | 8.51 | 2.39 | — | 9.73 | 0 | — | -0.808 |
| c.work_core_ms | 502 | 510 | 94.7 | 461 | 27.7 | 89.4 | 0.769 |
| gpu_temp_0 | 0.274 | 0.326 | 0.144 | 0.191 | 0.24 | 0.03 | 0.997 |
| gpu_temp_1 | 0.742 | 0.392 | 0.182 | 0.679 | 0.25 | 0.04 | 0.989 |

## dagger_r1 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 4.8 | 2.41 | 3.89 | 2.61 | 0.358 | 0.59 | 0.570 |
| a.queue_work_ms | 21 | 11.1 | 9.92 | 2.06 | 1.64 | 1.16 | -1.176 |
| a.cpu_cores | 0.53 | 0.239 | — | 0.296 | 0.111 | — | 0.656 |
| b.rate | 89.6 | 42.6 | 57.4 | 66.9 | 5.93 | 50.7 | 0.836 |
| b.buffer | 4.31 | 1.51 | — | 2.44 | 0 | — | 0.225 |
| c.work_core_ms | 238 | 173 | 83.4 | 119 | 27.5 | 44.5 | 0.785 |
| gpu_temp_0 | 0.17 | 0.342 | 0.25 | 0.0719 | 0.26 | 0.17 | 0.999 |
| gpu_temp_1 | 0.375 | 0.398 | 0.209 | 0.266 | 0.31 | 0.15 | 0.998 |

Overloaded states only (A queue > 100 ms): 18 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 42.3 | 19.4 | 82.6 | 33 | 8.51 | 79.6 | 0.440 |
| a.queue_work_ms | 344 | 290 | 206 | 113 | 230 | 155 | -1.490 |
| a.cpu_cores | 1.37 | 0.898 | — | 1.38 | 0.527 | — | -11.457 |
| b.rate | 30.6 | 314 | 12.1 | 17.3 | 233 | 1.7 | 0.787 |
| b.buffer | 8.46 | 6.06 | — | 9.75 | 0 | — | -0.960 |
| c.work_core_ms | 551 | 765 | 56.6 | 565 | 994 | 28.9 | 0.608 |
| gpu_temp_0 | 0.415 | 0.457 | 0.0817 | 0.198 | 0.495 | 0.02 | 0.995 |
| gpu_temp_1 | 0.875 | 0.629 | 0.118 | 0.865 | 0.535 | 0.03 | 0.992 |

## dagger_r2 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 13.8 | 3.98 | 5.69 | 2.67 | 0.369 | 0.641 | 0.754 |
| a.queue_work_ms | 107 | 22.7 | 16.1 | 1.29 | 1.6 | 1.36 | 0.450 |
| a.cpu_cores | 0.404 | 0.173 | — | 0.161 | 0.0985 | — | 0.713 |
| b.rate | 53.8 | 26.3 | 57 | 30.7 | 4.32 | 60.1 | 0.893 |
| b.buffer | 3.66 | 1.84 | — | 2.87 | 0.0676 | — | 0.494 |
| c.work_core_ms | 196 | 92.4 | 83.8 | 132 | 26.4 | 57.8 | 0.849 |
| gpu_temp_0 | 0.107 | 0.294 | 0.219 | 0.0608 | 0.13 | 0.15 | 1.000 |
| gpu_temp_1 | 0.279 | 0.362 | 0.182 | 0.166 | 0.29 | 0.13 | 0.999 |

Overloaded states only (A queue > 100 ms): 61 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 211 | 52.4 | 268 | 146 | 36 | 114 | 0.403 |
| a.queue_work_ms | 1.97e+03 | 378 | 253 | 1.75e+03 | 270 | 187 | -0.263 |
| a.cpu_cores | 1.41 | 0.224 | — | 1.36 | 0.0373 | — | -17.711 |
| b.rate | 1.33 | 33.9 | 4.51 | 0 | 0 | 0 | 0.994 |
| b.buffer | 12.5 | 1.3 | — | 12.9 | 0 | — | -12.899 |
| c.work_core_ms | 719 | 151 | 110 | 701 | 2.34 | 115 | -2.158 |
| gpu_temp_0 | 0.0881 | 0.07 | 0.0244 | 0.0409 | 0.01 | 0.02 | 1.000 |
| gpu_temp_1 | 0.658 | 0.203 | 0.0444 | 0.601 | 0.2 | 0.02 | 0.992 |
