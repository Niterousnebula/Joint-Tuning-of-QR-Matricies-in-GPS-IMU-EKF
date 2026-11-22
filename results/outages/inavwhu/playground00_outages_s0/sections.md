**inavwhu/playground00_outages_s0, section 1: 5 m GPS noise (76.2 m, 61 s), GPS error 4.87 m**

| Run | RMSE_h in section (m) | Max (m) | At end (m) | Next 30 s (m) | Recovers in (s) | Mean NIS in section | RMSE_h outside sections (m) | RMSE_h whole drive (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| THETA0 (baseline) | 2.81 | 6.74 | 4.70 | 1.37 | 5 | 47.44 | 1.12 | 6.582 |
| GA best (seed 3) | 1.96 | 4.85 | 3.08 | 1.42 | 10 | 7.34 | 1.16 | 1.828 |
| PSO best (seed 2) | 1.95 | 4.77 | 3.07 | 1.44 | 10 | 7.22 | 1.18 | 1.813 |
| DE best (seed 3) | 1.85 | 4.32 | 2.46 | 1.40 | 11 | 7.62 | 1.29 | 1.802 |
| GWO best (seed 5) | 1.89 | 4.50 | 2.72 | 1.39 | 10 | 8.45 | 1.22 | 1.810 |

**inavwhu/playground00_outages_s0, section 2: 15 m GPS noise (76.2 m, 52 s), GPS error 17.19 m**

| Run | RMSE_h in section (m) | Max (m) | At end (m) | Next 30 s (m) | Recovers in (s) | Mean NIS in section | RMSE_h outside sections (m) | RMSE_h whole drive (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| THETA0 (baseline) | 9.97 | 21.23 | 6.66 | 3.75 | 7 | 392.39 | 1.12 | 6.582 |
| GA best (seed 3) | 6.36 | 12.71 | 8.13 | 3.34 | 11 | 65.12 | 1.16 | 1.828 |
| PSO best (seed 2) | 6.31 | 12.65 | 8.17 | 3.33 | 11 | 65.34 | 1.18 | 1.813 |
| DE best (seed 3) | 5.74 | 10.88 | 8.67 | 3.62 | 12 | 58.46 | 1.29 | 1.802 |
| GWO best (seed 5) | 6.01 | 11.86 | 8.55 | 3.49 | 12 | 65.44 | 1.22 | 1.810 |

**inavwhu/playground00_outages_s0, section 3: GPS outage (76.2 m, 54 s), no fixes**

| Run | RMSE_h in section (m) | Max (m) | At end (m) | Next 30 s (m) | Recovers in (s) | Mean NIS in section | RMSE_h outside sections (m) | RMSE_h whole drive (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| THETA0 (baseline) | 28.75 | 71.43 | 71.43 | 13.49 | 70 | – | 1.12 | 6.582 |
| GA best (seed 3) | 1.75 | 3.80 | 3.80 | 1.77 | 75 | – | 1.16 | 1.828 |
| PSO best (seed 2) | 1.27 | 1.45 | 0.88 | 1.59 | 0 | – | 1.18 | 1.813 |
| DE best (seed 3) | 0.99 | 1.33 | 1.01 | 1.72 | 0 | – | 1.29 | 1.802 |
| GWO best (seed 5) | 1.55 | 3.21 | 3.21 | 1.73 | 74 | – | 1.22 | 1.810 |
