# One-step evaluation — `models/mlp_v1.pt`

MAE in real units (ms, batches/s, core-ms per window, °C). Lower is better.
Noise floor = the error even a perfect model would have (simulator randomness, measured on adaptive test runs).

## adaptive (3570 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² | noise floor |
|---|---|---|---|---|---|
| a.p99_ms | 5.1 | 6.36 | 8.67 | 0.891 | 0.388 |
| a.queue_work_ms | 74.5 | 31.7 | 21.8 | -0.608 | 1.18 |
| a.cpu_cores | 0.192 | 0.207 | — | 0.943 | 0.0787 |
| b.rate | 18.5 | 96.8 | 95 | 0.964 | 1.2 |
| b.buffer | 1.74 | 2.97 | — | 0.744 | 0.204 |
| c.work_core_ms | 49.9 | 349 | 31.9 | 0.989 | 16.2 |
| gpu_temp_0 | 0.0372 | 0.309 | 0.213 | 1.000 | 0.000265 |
| gpu_temp_1 | 0.077 | 0.415 | 0.148 | 1.000 | 0.00457 |

Overloaded states only (top 5% by A's queue):

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² |
|---|---|---|---|---|
| a.p99_ms | 75.8 | 68.9 | 166 | 0.813 |
| a.queue_work_ms | 1.36e+03 | 507 | 322 | -1.523 |
| a.cpu_cores | 0.243 | 0.417 | — | 0.834 |
| b.rate | 36.9 | 89.8 | 195 | 0.845 |
| b.buffer | 1.64 | 2.67 | — | 0.504 |
| c.work_core_ms | 56.8 | 253 | 30.5 | 0.962 |
| gpu_temp_0 | 0.0718 | 0.295 | 0.86 | 1.000 |
| gpu_temp_1 | 0.139 | 0.691 | 0.257 | 1.000 |

## mpc (5950 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² | noise floor |
|---|---|---|---|---|---|
| a.p99_ms | 38.8 | 5.19 | 6.53 | -15.317 | 0.388 |
| a.queue_work_ms | 1.07e+04 | 24.2 | 20.5 | -195811.235 | 1.18 |
| a.cpu_cores | 0.476 | 0.2 | — | 0.698 | 0.0787 |
| b.rate | 107 | 58.7 | 61.5 | 0.658 | 1.2 |
| b.buffer | 5.37 | 2.81 | — | -0.105 | 0.204 |
| c.work_core_ms | 252 | 251 | 90 | 0.820 | 16.2 |
| gpu_temp_0 | 0.2 | 0.326 | 0.176 | 0.999 | 0.000265 |
| gpu_temp_1 | 0.387 | 0.381 | 0.148 | 0.997 | 0.00457 |

Overloaded states only (top 5% by A's queue):

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² |
|---|---|---|---|---|
| a.p99_ms | 653 | 58.3 | 129 | -23.483 |
| a.queue_work_ms | 2.09e+05 | 365 | 301 | -270846.516 |
| a.cpu_cores | 0.751 | 0.517 | — | -0.560 |
| b.rate | 150 | 134 | 27.2 | 0.660 |
| b.buffer | 5.3 | 2.42 | — | 0.088 |
| c.work_core_ms | 368 | 505 | 102 | 0.865 |
| gpu_temp_0 | 0.239 | 0.318 | 0.143 | 0.998 |
| gpu_temp_1 | 0.67 | 0.386 | 0.188 | 0.991 |

## dagger_r1 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² | noise floor |
|---|---|---|---|---|---|
| a.p99_ms | 3.68 | 2.41 | 3.89 | 0.448 | 0.388 |
| a.queue_work_ms | 19.3 | 11.1 | 9.92 | -2.668 | 1.18 |
| a.cpu_cores | 0.359 | 0.239 | — | 0.835 | 0.0787 |
| b.rate | 90.1 | 42.6 | 57.4 | 0.856 | 1.2 |
| b.buffer | 4.61 | 1.51 | — | 0.295 | 0.204 |
| c.work_core_ms | 182 | 173 | 83.4 | 0.922 | 16.2 |
| gpu_temp_0 | 0.12 | 0.342 | 0.25 | 1.000 | 0.000265 |
| gpu_temp_1 | 0.417 | 0.398 | 0.209 | 0.997 | 0.00457 |

Overloaded states only (top 5% by A's queue):

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² |
|---|---|---|---|---|
| a.p99_ms | 24.5 | 10.9 | 34.5 | 0.385 |
| a.queue_work_ms | 256 | 117 | 90.9 | -3.936 |
| a.cpu_cores | 0.794 | 0.641 | — | 0.498 |
| b.rate | 109 | 166 | 38.9 | 0.892 |
| b.buffer | 6.87 | 3.78 | — | -0.501 |
| c.work_core_ms | 384 | 552 | 91.2 | 0.870 |
| gpu_temp_0 | 0.233 | 0.414 | 0.095 | 0.999 |
| gpu_temp_1 | 0.817 | 0.504 | 0.161 | 0.992 |

## dagger_r2 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² | noise floor |
|---|---|---|---|---|---|
| a.p99_ms | 21.6 | 3.98 | 5.69 | 0.333 | 0.388 |
| a.queue_work_ms | 65.3 | 22.7 | 16.1 | 0.698 | 1.18 |
| a.cpu_cores | 0.264 | 0.173 | — | 0.890 | 0.0787 |
| b.rate | 64.5 | 26.3 | 57 | 0.891 | 1.2 |
| b.buffer | 4.13 | 1.84 | — | 0.414 | 0.204 |
| c.work_core_ms | 148 | 92.4 | 83.8 | 0.889 | 16.2 |
| gpu_temp_0 | 0.0774 | 0.294 | 0.219 | 1.000 | 0.000265 |
| gpu_temp_1 | 0.286 | 0.362 | 0.182 | 0.998 | 0.00457 |

Overloaded states only (top 5% by A's queue):

| target | mlp MAE | persistence MAE | fluid MAE | mlp R² |
|---|---|---|---|---|
| a.p99_ms | 388 | 53.3 | 282 | -0.661 |
| a.queue_work_ms | 1.2e+03 | 383 | 257 | 0.288 |
| a.cpu_cores | 0.839 | 0.209 | — | -6.041 |
| b.rate | 132 | 34.4 | 4.58 | -1.981 |
| b.buffer | 13.8 | 1.32 | — | -16.427 |
| c.work_core_ms | 861 | 152 | 109 | -3.798 |
| gpu_temp_0 | 0.277 | 0.0653 | 0.0242 | 0.998 |
| gpu_temp_1 | 0.97 | 0.206 | 0.0443 | 0.985 |
