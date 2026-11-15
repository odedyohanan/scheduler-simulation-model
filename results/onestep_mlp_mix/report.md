# One-step evaluation — `models/mlp_mix.pt`

Mean and median absolute error in real units (ms, batches/s, core-ms per window, °C). Lower is better. A mean far above the median means a few huge misses dominate.

*Noise floor: the error left even when the simulator's full hidden state is known (queued requests, burst schedule), measured on 50 adaptive test states by re-simulating each window with 20 seeds. A model that sees only telemetry can't reach it; it is a lower bound, not a target.

## adaptive (3570 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 4.33 | 6.36 | 8.67 | 0.437 | 0.674 | 0.781 | 0.930 |
| a.queue_work_ms | 104 | 31.7 | 21.8 | 1.25 | 1.65 | 1.22 | -4.416 |
| a.cpu_cores | 0.179 | 0.207 | — | 0.111 | 0.11 | — | 0.946 |
| b.rate | 17.8 | 96.8 | 95 | 9.33 | 5.98 | 95.6 | 0.966 |
| b.buffer | 1.48 | 2.97 | — | 0.581 | 0.379 | — | 0.771 |
| c.work_core_ms | 41.1 | 349 | 31.9 | 28.3 | 41.2 | 16.3 | 0.992 |
| gpu_temp_0 | 0.0264 | 0.309 | 0.213 | 0.0168 | 0.15 | 0.14 | 1.000 |
| gpu_temp_1 | 0.06 | 0.415 | 0.148 | 0.026 | 0.33 | 0.13 | 1.000 |

Overloaded states only (A queue > 100 ms): 174 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 63.7 | 70.7 | 172 | 26.4 | 35 | 84.4 | 0.880 |
| a.queue_work_ms | 2.03e+03 | 519 | 330 | 427 | 317 | 219 | -7.668 |
| a.cpu_cores | 0.199 | 0.416 | — | 0.122 | 0.0945 | — | 0.871 |
| b.rate | 37.5 | 90.3 | 197 | 26.4 | 37.5 | 193 | 0.820 |
| b.buffer | 1.7 | 2.72 | — | 0.696 | 0 | — | 0.556 |
| c.work_core_ms | 59.1 | 259 | 31 | 50.8 | 15.2 | 25 | 0.973 |
| gpu_temp_0 | 0.0621 | 0.286 | 0.867 | 0.0508 | 0.2 | 0.98 | 1.000 |
| gpu_temp_1 | 0.131 | 0.696 | 0.256 | 0.0868 | 0.73 | 0.2 | 1.000 |

## mpc (5950 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 3.64 | 5.19 | 6.53 | 0.464 | 0.518 | 0.911 | 0.955 |
| a.queue_work_ms | 63.1 | 24.2 | 20.5 | 1.35 | 1.74 | 1.35 | -1.566 |
| a.cpu_cores | 0.176 | 0.2 | — | 0.11 | 0.11 | — | 0.932 |
| b.rate | 19.1 | 58.7 | 61.5 | 11.4 | 6.85 | 61.3 | 0.983 |
| b.buffer | 2.13 | 2.81 | — | 1.22 | 0.357 | — | 0.770 |
| c.work_core_ms | 67.8 | 251 | 90 | 49.1 | 46.4 | 65.2 | 0.984 |
| gpu_temp_0 | 0.0313 | 0.326 | 0.176 | 0.022 | 0.22 | 0.14 | 1.000 |
| gpu_temp_1 | 0.0699 | 0.381 | 0.148 | 0.043 | 0.31 | 0.12 | 1.000 |

Overloaded states only (A queue > 100 ms): 242 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 56.5 | 66.4 | 163 | 23.5 | 31.5 | 91.2 | 0.933 |
| a.queue_work_ms | 1.4e+03 | 430 | 354 | 177 | 294 | 299 | -2.871 |
| a.cpu_cores | 0.219 | 0.465 | — | 0.138 | 0.0838 | — | 0.759 |
| b.rate | 36.6 | 132 | 25.7 | 19.8 | 0.876 | 6.35 | 0.965 |
| b.buffer | 1.66 | 2.39 | — | 1.39 | 0 | — | 0.914 |
| c.work_core_ms | 91.3 | 510 | 94.7 | 77 | 27.7 | 89.4 | 0.990 |
| gpu_temp_0 | 0.0755 | 0.326 | 0.144 | 0.0596 | 0.24 | 0.03 | 1.000 |
| gpu_temp_1 | 0.131 | 0.392 | 0.182 | 0.0848 | 0.25 | 0.04 | 0.999 |

## dagger_r1 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 1.61 | 2.41 | 3.89 | 0.393 | 0.358 | 0.59 | 0.800 |
| a.queue_work_ms | 7.93 | 11.1 | 9.92 | 1.11 | 1.64 | 1.16 | 0.531 |
| a.cpu_cores | 0.218 | 0.239 | — | 0.114 | 0.111 | — | 0.909 |
| b.rate | 18.9 | 42.6 | 57.4 | 8.94 | 5.93 | 50.7 | 0.981 |
| b.buffer | 1.31 | 1.51 | — | 0.699 | 0 | — | 0.889 |
| c.work_core_ms | 59.9 | 173 | 83.4 | 37 | 27.5 | 44.5 | 0.984 |
| gpu_temp_0 | 0.0336 | 0.342 | 0.25 | 0.0218 | 0.26 | 0.17 | 1.000 |
| gpu_temp_1 | 0.0679 | 0.398 | 0.209 | 0.0313 | 0.31 | 0.15 | 1.000 |

Overloaded states only (A queue > 100 ms): 18 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 22.9 | 19.4 | 82.6 | 21.9 | 8.51 | 79.6 | 0.876 |
| a.queue_work_ms | 174 | 290 | 206 | 58.8 | 230 | 155 | 0.479 |
| a.cpu_cores | 0.253 | 0.898 | — | 0.23 | 0.527 | — | 0.517 |
| b.rate | 18.3 | 314 | 12.1 | 18.2 | 233 | 1.7 | 0.943 |
| b.buffer | 2.17 | 6.06 | — | 2 | 0 | — | 0.899 |
| c.work_core_ms | 78.4 | 765 | 56.6 | 58.2 | 994 | 28.9 | 0.990 |
| gpu_temp_0 | 0.125 | 0.457 | 0.0817 | 0.119 | 0.495 | 0.02 | 1.000 |
| gpu_temp_1 | 0.105 | 0.629 | 0.118 | 0.0603 | 0.535 | 0.03 | 1.000 |

## dagger_r2 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 2.17 | 3.98 | 5.69 | 0.357 | 0.369 | 0.641 | 0.994 |
| a.queue_work_ms | 70.5 | 22.7 | 16.1 | 1.16 | 1.6 | 1.36 | 0.468 |
| a.cpu_cores | 0.156 | 0.173 | — | 0.104 | 0.0985 | — | 0.952 |
| b.rate | 16.5 | 26.3 | 57 | 9.7 | 4.32 | 60.1 | 0.985 |
| b.buffer | 1.39 | 1.84 | — | 0.77 | 0.0676 | — | 0.889 |
| c.work_core_ms | 55 | 92.4 | 83.8 | 38.1 | 26.4 | 57.8 | 0.987 |
| gpu_temp_0 | 0.0261 | 0.294 | 0.219 | 0.0196 | 0.13 | 0.15 | 1.000 |
| gpu_temp_1 | 0.0555 | 0.362 | 0.182 | 0.032 | 0.29 | 0.13 | 1.000 |

Overloaded states only (A queue > 100 ms): 61 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 30.8 | 52.4 | 268 | 20.7 | 36 | 114 | 0.985 |
| a.queue_work_ms | 1.33e+03 | 378 | 253 | 286 | 270 | 187 | -0.227 |
| a.cpu_cores | 0.15 | 0.224 | — | 0.0973 | 0.0373 | — | 0.472 |
| b.rate | 17.7 | 33.9 | 4.51 | 19.2 | 0 | 0 | 0.925 |
| b.buffer | 0.497 | 1.3 | — | 0.191 | 0 | — | 0.934 |
| c.work_core_ms | 103 | 151 | 110 | 88.1 | 2.34 | 115 | 0.916 |
| gpu_temp_0 | 0.068 | 0.07 | 0.0244 | 0.0716 | 0.01 | 0.02 | 1.000 |
| gpu_temp_1 | 0.0613 | 0.203 | 0.0444 | 0.0549 | 0.2 | 0.02 | 1.000 |
