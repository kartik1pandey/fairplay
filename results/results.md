| Experiment | Split | Claim | Value | 95% CI | Run |
|---|---|---|---|---|---|
| R | test (Aug) | flag-precision@k (c = 0.5%), LightGBM | 0.388 | 0.360 – 0.415 | `20261030T073356Z_final` |
| R | test (Aug) | flag-precision@k (c = 0.5%), logistic regression | 0.206 | 0.183 – 0.226 | `20261030T073356Z_final` |
| R | test (Aug) | flag-precision@k (c = 0.5%), rule baseline | 0.158 | 0.141 – 0.179 | `20261030T073356Z_final` |
| R | test (Aug) | lift@k, LightGBM (× background rate) | 23.115 | 21.327 – 24.916 | `20261030T073356Z_final` |
| R | test (Aug) | flag-recall@k, LightGBM | 0.339 | 0.319 – 0.360 | `20261030T073356Z_final` |
| R | test (Aug) | flag-PR-AUC, LightGBM | 0.379 | 0.355 – 0.402 | `20261030T073356Z_final` |
| R | test (Aug) | LightGBM − rule, flag-precision (pre-registered) | 0.230 | 0.204 – 0.252 | `20261030T073356Z_final` |
| R | test (Aug) | LightGBM − logistic regression, flag-precision (pre-registered) | 0.182 | 0.159 – 0.207 | `20261030T073356Z_final` |
| R | test, unseen | flag-precision@k, LightGBM, unseen accounts | 0.769 | 0.689 – 0.832 | `20261030T073356Z_final` |
| R | test, unseen | LightGBM − rule, unseen accounts (pre-registered) | 0.496 | 0.407 – 0.569 | `20261030T073356Z_final` |
| R | test, unseen | LightGBM − logistic regression, unseen accounts (pre-registered) | 0.353 | 0.277 – 0.426 | `20261030T073356Z_final` |
| R | test (Aug) | Brier score, LightGBM (calibrated) | 0.017 | – | `20261030T073356Z_final` |
| R | test (Aug) | flag-precision@k with the 2200+ cap, LightGBM | 0.372 | – | `20261030T073356Z_final` |
| R | test (Aug) | selection rate among unflagged, accounts < 30 days ÷ overall (LightGBM, capped) | 4.393 | – | `20261030T073356Z_final` |
| R | test (Aug) | selection rate among unflagged, accounts 30–365 days ÷ overall (LightGBM, capped) | 1.628 | – | `20261030T073356Z_final` |
| R | test (Aug) | selection rate among unflagged, fast improvers ÷ overall (LightGBM, capped) | 2.356 | – | `20261030T073356Z_final` |
| R | test (Aug) | selection rate among unflagged, 2200+ rated ÷ overall (LightGBM, capped) | 0.548 | – | `20261030T073356Z_final` |
