# One-step evaluation — `models/mlp_v1.pt`

Mean and median absolute error in real units (ms, batches/s, core-ms per window, °C). Lower is better. A mean far above the median means a few huge misses dominate.

*Noise floor: the error left even when the simulator's full hidden state is known (queued requests, burst schedule), measured on 50 adaptive test states by re-simulating each window with 20 seeds. A model that sees only telemetry can't reach it; it is a lower bound, not a target.

## adaptive (3570 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² | noise floor* |
|---|---|---|---|---|---|---|---|---|
| a.p99_ms | 5.1 | 6.36 | 8.67 | 0.496 | 0.674 | 0.781 | 0.891 | 0.388 |
| a.queue_work_ms | 74.5 | 31.7 | 21.8 | 1.23 | 1.65 | 1.22 | -0.608 | 1.18 |
| a.cpu_cores | 0.192 | 0.207 | — | 0.125 | 0.11 | — | 0.943 | 0.0787 |
| b.rate | 18.5 | 96.8 | 95 | 8.72 | 5.98 | 95.6 | 0.964 | 1.2 |
| b.buffer | 1.74 | 2.97 | — | 0.835 | 0.379 | — | 0.744 | 0.204 |
| c.work_core_ms | 49.9 | 349 | 31.9 | 35.4 | 41.2 | 16.3 | 0.989 | 16.2 |
| gpu_temp_0 | 0.0372 | 0.309 | 0.213 | 0.0243 | 0.15 | 0.14 | 1.000 | 0.000265 |
| gpu_temp_1 | 0.077 | 0.415 | 0.148 | 0.0436 | 0.33 | 0.13 | 1.000 | 0.00457 |

Overloaded states only (A queue > 100 ms): 174 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 77.8 | 70.7 | 172 | 25.2 | 35 | 84.4 | 0.810 |
| a.queue_work_ms | 1.4e+03 | 519 | 330 | 413 | 317 | 219 | -1.566 |
| a.cpu_cores | 0.243 | 0.416 | — | 0.16 | 0.0945 | — | 0.832 |
| b.rate | 37.2 | 90.3 | 197 | 28 | 37.5 | 193 | 0.833 |
| b.buffer | 1.66 | 2.72 | — | 0.033 | 0 | — | 0.505 |
| c.work_core_ms | 57.1 | 259 | 31 | 35.9 | 15.2 | 25 | 0.962 |
| gpu_temp_0 | 0.0727 | 0.286 | 0.867 | 0.0581 | 0.2 | 0.98 | 1.000 |
| gpu_temp_1 | 0.141 | 0.696 | 0.256 | 0.0896 | 0.73 | 0.2 | 1.000 |

## mpc (5950 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² | noise floor* |
|---|---|---|---|---|---|---|---|---|
| a.p99_ms | 38.8 | 5.19 | 6.53 | 2.25 | 0.518 | 0.911 | -15.317 | 0.388 |
| a.queue_work_ms | 1.07e+04 | 24.2 | 20.5 | 2.96 | 1.74 | 1.35 | -195811.235 | 1.18 |
| a.cpu_cores | 0.476 | 0.2 | — | 0.354 | 0.11 | — | 0.698 | 0.0787 |
| b.rate | 107 | 58.7 | 61.5 | 80.1 | 6.85 | 61.3 | 0.658 | 1.2 |
| b.buffer | 5.37 | 2.81 | — | 4.13 | 0.357 | — | -0.105 | 0.204 |
| c.work_core_ms | 252 | 251 | 90 | 197 | 46.4 | 65.2 | 0.820 | 16.2 |
| gpu_temp_0 | 0.2 | 0.326 | 0.176 | 0.168 | 0.22 | 0.14 | 0.999 | 0.000265 |
| gpu_temp_1 | 0.387 | 0.381 | 0.148 | 0.295 | 0.31 | 0.12 | 0.997 | 0.00457 |

Overloaded states only (A queue > 100 ms): 242 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 800 | 66.4 | 163 | 93.6 | 31.5 | 91.2 | -26.324 |
| a.queue_work_ms | 2.58e+05 | 430 | 354 | 1.11e+03 | 294 | 299 | -295635.963 |
| a.cpu_cores | 0.74 | 0.465 | — | 0.561 | 0.0838 | — | -0.984 |
| b.rate | 153 | 132 | 25.7 | 139 | 0.876 | 6.35 | 0.606 |
| b.buffer | 5.17 | 2.39 | — | 3.24 | 0 | — | 0.117 |
| c.work_core_ms | 380 | 510 | 94.7 | 314 | 27.7 | 89.4 | 0.853 |
| gpu_temp_0 | 0.237 | 0.326 | 0.144 | 0.146 | 0.24 | 0.03 | 0.998 |
| gpu_temp_1 | 0.677 | 0.392 | 0.182 | 0.698 | 0.25 | 0.04 | 0.991 |

## dagger_r1 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² | noise floor* |
|---|---|---|---|---|---|---|---|---|
| a.p99_ms | 3.68 | 2.41 | 3.89 | 1.54 | 0.358 | 0.59 | 0.448 | 0.388 |
| a.queue_work_ms | 19.3 | 11.1 | 9.92 | 1.68 | 1.64 | 1.16 | -2.668 | 1.18 |
| a.cpu_cores | 0.359 | 0.239 | — | 0.212 | 0.111 | — | 0.835 | 0.0787 |
| b.rate | 90.1 | 42.6 | 57.4 | 80.3 | 5.93 | 50.7 | 0.856 | 1.2 |
| b.buffer | 4.61 | 1.51 | — | 3.91 | 0 | — | 0.295 | 0.204 |
| c.work_core_ms | 182 | 173 | 83.4 | 156 | 27.5 | 44.5 | 0.922 | 16.2 |
| gpu_temp_0 | 0.12 | 0.342 | 0.25 | 0.0602 | 0.26 | 0.17 | 1.000 | 0.000265 |
| gpu_temp_1 | 0.417 | 0.398 | 0.209 | 0.31 | 0.31 | 0.15 | 0.997 | 0.00457 |

Overloaded states only (A queue > 100 ms): 18 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 59 | 19.4 | 82.6 | 36.3 | 8.51 | 79.6 | -0.195 |
| a.queue_work_ms | 742 | 290 | 206 | 448 | 230 | 155 | -5.679 |
| a.cpu_cores | 0.615 | 0.898 | — | 0.435 | 0.527 | — | -1.592 |
| b.rate | 91.8 | 314 | 12.1 | 66.3 | 233 | 1.7 | -0.315 |
| b.buffer | 4.4 | 6.06 | — | 4.82 | 0 | — | 0.550 |
| c.work_core_ms | 474 | 765 | 56.6 | 463 | 994 | 28.9 | 0.752 |
| gpu_temp_0 | 0.331 | 0.457 | 0.0817 | 0.293 | 0.495 | 0.02 | 0.998 |
| gpu_temp_1 | 0.983 | 0.629 | 0.118 | 0.992 | 0.535 | 0.03 | 0.992 |

## dagger_r2 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² | noise floor* |
|---|---|---|---|---|---|---|---|---|
| a.p99_ms | 21.6 | 3.98 | 5.69 | 1.23 | 0.369 | 0.641 | 0.333 | 0.388 |
| a.queue_work_ms | 65.3 | 22.7 | 16.1 | 1.23 | 1.6 | 1.36 | 0.698 | 1.18 |
| a.cpu_cores | 0.264 | 0.173 | — | 0.162 | 0.0985 | — | 0.890 | 0.0787 |
| b.rate | 64.5 | 26.3 | 57 | 50.1 | 4.32 | 60.1 | 0.891 | 1.2 |
| b.buffer | 4.13 | 1.84 | — | 3.48 | 0.0676 | — | 0.414 | 0.204 |
| c.work_core_ms | 148 | 92.4 | 83.8 | 88.3 | 26.4 | 57.8 | 0.889 | 16.2 |
| gpu_temp_0 | 0.0774 | 0.294 | 0.219 | 0.0414 | 0.13 | 0.15 | 1.000 | 0.000265 |
| gpu_temp_1 | 0.286 | 0.362 | 0.182 | 0.19 | 0.29 | 0.13 | 0.998 | 0.00457 |

Overloaded states only (A queue > 100 ms): 61 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 381 | 52.4 | 268 | 329 | 36 | 114 | -0.622 |
| a.queue_work_ms | 1.18e+03 | 378 | 253 | 843 | 270 | 187 | 0.305 |
| a.cpu_cores | 0.83 | 0.224 | — | 0.892 | 0.0373 | — | -5.443 |
| b.rate | 132 | 33.9 | 4.51 | 147 | 0 | 0 | -2.027 |
| b.buffer | 13.6 | 1.3 | — | 16 | 0 | — | -16.436 |
| c.work_core_ms | 852 | 151 | 110 | 1.03e+03 | 2.34 | 115 | -3.805 |
| gpu_temp_0 | 0.275 | 0.07 | 0.0244 | 0.28 | 0.01 | 0.02 | 0.998 |
| gpu_temp_1 | 0.969 | 0.203 | 0.0444 | 0.952 | 0.2 | 0.02 | 0.986 |
