# One-step evaluation — `models/gru_mix.pt`

Mean and median absolute error in real units (ms, batches/s, core-ms per window, °C). Lower is better. A mean far above the median means a few huge misses dominate.

*Noise floor: the error left even when the simulator's full hidden state is known (queued requests, burst schedule), measured on 50 adaptive test states by re-simulating each window with 20 seeds. A model that sees only telemetry can't reach it; it is a lower bound, not a target.

## adaptive (3570 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 5.37 | 6.36 | 8.67 | 0.514 | 0.674 | 0.781 | 0.919 |
| a.queue_work_ms | 43.7 | 31.7 | 21.8 | 1.18 | 1.65 | 1.22 | 0.800 |
| a.cpu_cores | 0.182 | 0.207 | — | 0.114 | 0.11 | — | 0.945 |
| b.rate | 19.5 | 96.8 | 95 | 10.4 | 5.98 | 95.6 | 0.963 |
| b.buffer | 1.78 | 2.97 | — | 0.833 | 0.379 | — | 0.744 |
| c.work_core_ms | 47 | 349 | 31.9 | 32.7 | 41.2 | 16.3 | 0.990 |
| gpu_temp_0 | 0.0351 | 0.309 | 0.213 | 0.0252 | 0.15 | 0.14 | 1.000 |
| gpu_temp_1 | 0.0705 | 0.415 | 0.148 | 0.0399 | 0.33 | 0.13 | 1.000 |

Overloaded states only (A queue > 100 ms): 174 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 81.3 | 70.7 | 172 | 29.9 | 35 | 84.4 | 0.864 |
| a.queue_work_ms | 788 | 519 | 330 | 429 | 317 | 219 | 0.684 |
| a.cpu_cores | 0.237 | 0.416 | — | 0.153 | 0.0945 | — | 0.844 |
| b.rate | 42.1 | 90.3 | 197 | 29.7 | 37.5 | 193 | 0.802 |
| b.buffer | 2.74 | 2.72 | — | 2.09 | 0 | — | 0.449 |
| c.work_core_ms | 73.9 | 259 | 31 | 69.5 | 15.2 | 25 | 0.955 |
| gpu_temp_0 | 0.0748 | 0.286 | 0.867 | 0.0553 | 0.2 | 0.98 | 1.000 |
| gpu_temp_1 | 0.136 | 0.696 | 0.256 | 0.0903 | 0.73 | 0.2 | 1.000 |

## mpc (5950 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 4.93 | 5.19 | 6.53 | 0.59 | 0.518 | 0.911 | 0.932 |
| a.queue_work_ms | 48 | 24.2 | 20.5 | 1.27 | 1.74 | 1.35 | -0.029 |
| a.cpu_cores | 0.191 | 0.2 | — | 0.117 | 0.11 | — | 0.922 |
| b.rate | 24.5 | 58.7 | 61.5 | 15.5 | 6.85 | 61.3 | 0.975 |
| b.buffer | 2.42 | 2.81 | — | 1.63 | 0.357 | — | 0.746 |
| c.work_core_ms | 84.2 | 251 | 90 | 60.9 | 46.4 | 65.2 | 0.976 |
| gpu_temp_0 | 0.0434 | 0.326 | 0.176 | 0.0312 | 0.22 | 0.14 | 1.000 |
| gpu_temp_1 | 0.0885 | 0.381 | 0.148 | 0.0585 | 0.31 | 0.12 | 1.000 |

Overloaded states only (A queue > 100 ms): 242 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 83.8 | 66.4 | 163 | 35.9 | 31.5 | 91.2 | 0.895 |
| a.queue_work_ms | 1.04e+03 | 430 | 354 | 258 | 294 | 299 | -0.547 |
| a.cpu_cores | 0.267 | 0.465 | — | 0.192 | 0.0838 | — | 0.718 |
| b.rate | 37.3 | 132 | 25.7 | 17 | 0.876 | 6.35 | 0.958 |
| b.buffer | 1.64 | 2.39 | — | 1.09 | 0 | — | 0.904 |
| c.work_core_ms | 120 | 510 | 94.7 | 81.3 | 27.7 | 89.4 | 0.981 |
| gpu_temp_0 | 0.0972 | 0.326 | 0.144 | 0.0652 | 0.24 | 0.03 | 1.000 |
| gpu_temp_1 | 0.147 | 0.392 | 0.182 | 0.0859 | 0.25 | 0.04 | 0.999 |

## dagger_r1 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 1.87 | 2.41 | 3.89 | 0.415 | 0.358 | 0.59 | 0.742 |
| a.queue_work_ms | 7.94 | 11.1 | 9.92 | 0.998 | 1.64 | 1.16 | 0.604 |
| a.cpu_cores | 0.206 | 0.239 | — | 0.111 | 0.111 | — | 0.918 |
| b.rate | 23.3 | 42.6 | 57.4 | 13.2 | 5.93 | 50.7 | 0.979 |
| b.buffer | 1.3 | 1.51 | — | 0.63 | 0 | — | 0.896 |
| c.work_core_ms | 62 | 173 | 83.4 | 39.7 | 27.5 | 44.5 | 0.983 |
| gpu_temp_0 | 0.0388 | 0.342 | 0.25 | 0.0219 | 0.26 | 0.17 | 1.000 |
| gpu_temp_1 | 0.076 | 0.398 | 0.209 | 0.0431 | 0.31 | 0.15 | 1.000 |

Overloaded states only (A queue > 100 ms): 18 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 31.4 | 19.4 | 82.6 | 23.7 | 8.51 | 79.6 | 0.694 |
| a.queue_work_ms | 157 | 290 | 206 | 75.1 | 230 | 155 | 0.669 |
| a.cpu_cores | 0.232 | 0.898 | — | 0.158 | 0.527 | — | 0.528 |
| b.rate | 10.9 | 314 | 12.1 | 5.82 | 233 | 1.7 | 0.962 |
| b.buffer | 1.08 | 6.06 | — | 0.521 | 0 | — | 0.958 |
| c.work_core_ms | 127 | 765 | 56.6 | 95.9 | 994 | 28.9 | 0.974 |
| gpu_temp_0 | 0.111 | 0.457 | 0.0817 | 0.0737 | 0.495 | 0.02 | 1.000 |
| gpu_temp_1 | 0.114 | 0.629 | 0.118 | 0.0778 | 0.535 | 0.03 | 1.000 |

## dagger_r2 (1190 transitions)

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 8.61 | 3.98 | 5.69 | 0.358 | 0.369 | 0.641 | 0.885 |
| a.queue_work_ms | 58 | 22.7 | 16.1 | 1.1 | 1.6 | 1.36 | 0.844 |
| a.cpu_cores | 0.16 | 0.173 | — | 0.103 | 0.0985 | — | 0.951 |
| b.rate | 19.1 | 26.3 | 57 | 12.2 | 4.32 | 60.1 | 0.983 |
| b.buffer | 1.58 | 1.84 | — | 1.04 | 0.0676 | — | 0.879 |
| c.work_core_ms | 60.6 | 92.4 | 83.8 | 43.5 | 26.4 | 57.8 | 0.985 |
| gpu_temp_0 | 0.0281 | 0.294 | 0.219 | 0.0201 | 0.13 | 0.15 | 1.000 |
| gpu_temp_1 | 0.0653 | 0.362 | 0.182 | 0.0401 | 0.29 | 0.13 | 1.000 |

Overloaded states only (A queue > 100 ms): 61 transitions

| target | mlp MAE | persistence MAE | fluid MAE | mlp median | persistence median | fluid median | mlp R² |
|---|---|---|---|---|---|---|---|
| a.p99_ms | 154 | 52.4 | 268 | 123 | 36 | 114 | 0.719 |
| a.queue_work_ms | 1.08e+03 | 378 | 253 | 1.13e+03 | 270 | 187 | 0.642 |
| a.cpu_cores | 0.171 | 0.224 | — | 0.077 | 0.0373 | — | 0.316 |
| b.rate | 2.45 | 33.9 | 4.51 | 0 | 0 | 0 | 0.986 |
| b.buffer | 0.881 | 1.3 | — | 0.609 | 0 | — | 0.877 |
| c.work_core_ms | 63.4 | 151 | 110 | 41.2 | 2.34 | 115 | 0.944 |
| gpu_temp_0 | 0.0454 | 0.07 | 0.0244 | 0.0368 | 0.01 | 0.02 | 1.000 |
| gpu_temp_1 | 0.0572 | 0.203 | 0.0444 | 0.0409 | 0.2 | 0.02 | 1.000 |
