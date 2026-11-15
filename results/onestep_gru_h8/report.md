# One-step evaluation — `models/gru_h8.pt`

Mean and median absolute error in real units (ms, batches/s, core-ms per window, °C). Lower is better. A mean far above the median means a few huge misses dominate.

*Noise floor: the error left even when the simulator's full hidden state is known (queued requests, burst schedule), measured on 50 adaptive test states by re-simulating each window with 20 seeds. A model that sees only telemetry can't reach it; it is a lower bound, not a target.

## adaptive (3570 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 4.47 | 6.36 | 8.67 | 0.574 | 0.674 | 0.781 | 0.957 |
| a.queue_work_ms | 45.2 | 31.7 | 21.8 | 1.19 | 1.65 | 1.22 | 0.831 |
| a.cpu_cores | 0.199 | 0.207 | — | 0.125 | 0.11 | — | 0.939 |
| b.rate | 22.6 | 96.8 | 95 | 11.9 | 5.98 | 95.6 | 0.955 |
| b.buffer | 1.9 | 2.97 | — | 0.927 | 0.379 | — | 0.721 |
| c.work_core_ms | 56.2 | 349 | 31.9 | 37.5 | 41.2 | 16.3 | 0.985 |
| gpu_temp_0 | 0.0445 | 0.309 | 0.213 | 0.0309 | 0.15 | 0.14 | 1.000 |
| gpu_temp_1 | 0.0835 | 0.415 | 0.148 | 0.0459 | 0.33 | 0.13 | 1.000 |

Overloaded states only (A queue > 100 ms): 174 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 60.6 | 70.7 | 172 | 30.7 | 35 | 84.4 | 0.931 |
| a.queue_work_ms | 815 | 519 | 330 | 500 | 317 | 219 | 0.734 |
| a.cpu_cores | 0.29 | 0.416 | — | 0.228 | 0.0945 | — | 0.797 |
| b.rate | 48.3 | 90.3 | 197 | 35.2 | 37.5 | 193 | 0.768 |
| b.buffer | 2.12 | 2.72 | — | 0.575 | 0 | — | 0.462 |
| c.work_core_ms | 94 | 259 | 31 | 73.6 | 15.2 | 25 | 0.930 |
| gpu_temp_0 | 0.0958 | 0.286 | 0.867 | 0.0797 | 0.2 | 0.98 | 1.000 |
| gpu_temp_1 | 0.166 | 0.696 | 0.256 | 0.128 | 0.73 | 0.2 | 1.000 |

## mpc (5950 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 9.08 | 5.19 | 6.53 | 2.39 | 0.518 | 0.911 | 0.859 |
| a.queue_work_ms | 37.2 | 24.2 | 20.5 | 2.14 | 1.74 | 1.35 | 0.769 |
| a.cpu_cores | 0.304 | 0.2 | — | 0.229 | 0.11 | — | 0.865 |
| b.rate | 88.3 | 58.7 | 61.5 | 75 | 6.85 | 61.3 | 0.794 |
| b.buffer | 4.43 | 2.81 | — | 3.62 | 0.357 | — | 0.269 |
| c.work_core_ms | 260 | 251 | 90 | 195 | 46.4 | 65.2 | 0.788 |
| gpu_temp_0 | 0.189 | 0.326 | 0.176 | 0.151 | 0.22 | 0.14 | 0.999 |
| gpu_temp_1 | 0.377 | 0.381 | 0.148 | 0.336 | 0.31 | 0.12 | 0.998 |

Overloaded states only (A queue > 100 ms): 242 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 112 | 66.4 | 163 | 39.9 | 31.5 | 91.2 | 0.780 |
| a.queue_work_ms | 735 | 430 | 354 | 312 | 294 | 299 | 0.661 |
| a.cpu_cores | 0.424 | 0.465 | — | 0.261 | 0.0838 | — | 0.245 |
| b.rate | 84.9 | 132 | 25.7 | 37.7 | 0.876 | 6.35 | 0.801 |
| b.buffer | 7.59 | 2.39 | — | 6.29 | 0 | — | -0.729 |
| c.work_core_ms | 408 | 510 | 94.7 | 276 | 27.7 | 89.4 | 0.752 |
| gpu_temp_0 | 0.333 | 0.326 | 0.144 | 0.294 | 0.24 | 0.03 | 0.997 |
| gpu_temp_1 | 0.383 | 0.392 | 0.182 | 0.308 | 0.25 | 0.04 | 0.997 |

## dagger_r1 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 5.32 | 2.41 | 3.89 | 2.83 | 0.358 | 0.59 | 0.494 |
| a.queue_work_ms | 9.84 | 11.1 | 9.92 | 1.56 | 1.64 | 1.16 | 0.548 |
| a.cpu_cores | 0.332 | 0.239 | — | 0.248 | 0.111 | — | 0.871 |
| b.rate | 93.2 | 42.6 | 57.4 | 71.7 | 5.93 | 50.7 | 0.834 |
| b.buffer | 4.1 | 1.51 | — | 2.66 | 0 | — | 0.357 |
| c.work_core_ms | 234 | 173 | 83.4 | 193 | 27.5 | 44.5 | 0.829 |
| gpu_temp_0 | 0.145 | 0.342 | 0.25 | 0.0841 | 0.26 | 0.17 | 0.999 |
| gpu_temp_1 | 0.355 | 0.398 | 0.209 | 0.31 | 0.31 | 0.15 | 0.998 |

Overloaded states only (A queue > 100 ms): 18 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 38.2 | 19.4 | 82.6 | 20.8 | 8.51 | 79.6 | 0.450 |
| a.queue_work_ms | 197 | 290 | 206 | 120 | 230 | 155 | 0.584 |
| a.cpu_cores | 0.683 | 0.898 | — | 0.479 | 0.527 | — | -3.426 |
| b.rate | 176 | 314 | 12.1 | 76.1 | 233 | 1.7 | -6.525 |
| b.buffer | 7.98 | 6.06 | — | 8.69 | 0 | — | -0.760 |
| c.work_core_ms | 543 | 765 | 56.6 | 483 | 994 | 28.9 | 0.615 |
| gpu_temp_0 | 0.439 | 0.457 | 0.0817 | 0.439 | 0.495 | 0.02 | 0.997 |
| gpu_temp_1 | 0.667 | 0.629 | 0.118 | 0.559 | 0.535 | 0.03 | 0.996 |

## dagger_r2 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 14.8 | 3.98 | 5.69 | 2.4 | 0.369 | 0.641 | 0.756 |
| a.queue_work_ms | 62 | 22.7 | 16.1 | 1.27 | 1.6 | 1.36 | 0.812 |
| a.cpu_cores | 0.294 | 0.173 | — | 0.238 | 0.0985 | — | 0.899 |
| b.rate | 91 | 26.3 | 57 | 70.7 | 4.32 | 60.1 | 0.794 |
| b.buffer | 3.82 | 1.84 | — | 2.78 | 0.0676 | — | 0.460 |
| c.work_core_ms | 168 | 92.4 | 83.8 | 132 | 26.4 | 57.8 | 0.904 |
| gpu_temp_0 | 0.0922 | 0.294 | 0.219 | 0.0655 | 0.13 | 0.15 | 1.000 |
| gpu_temp_1 | 0.366 | 0.362 | 0.182 | 0.3 | 0.29 | 0.13 | 0.998 |

Overloaded states only (A queue > 100 ms): 61 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 226 | 52.4 | 268 | 199 | 36 | 114 | 0.410 |
| a.queue_work_ms | 1.15e+03 | 378 | 253 | 1.08e+03 | 270 | 187 | 0.566 |
| a.cpu_cores | 0.402 | 0.224 | — | 0.332 | 0.0373 | — | -1.182 |
| b.rate | 9.93 | 33.9 | 4.51 | 0 | 0 | 0 | 0.704 |
| b.buffer | 12.5 | 1.3 | — | 13 | 0 | — | -12.820 |
| c.work_core_ms | 396 | 151 | 110 | 355 | 2.34 | 115 | -0.375 |
| gpu_temp_0 | 0.291 | 0.07 | 0.0244 | 0.278 | 0.01 | 0.02 | 0.998 |
| gpu_temp_1 | 0.383 | 0.203 | 0.0444 | 0.385 | 0.2 | 0.02 | 0.998 |
