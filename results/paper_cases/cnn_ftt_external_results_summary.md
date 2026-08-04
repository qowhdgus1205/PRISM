# CNN-MLP and FT-Transformer External Results

Latest source: `results/uci_external_experiments/processed_extra_results.csv`. Metrics are averaged over 5 seeds. MSE is derived from seed-level target RMSE dictionaries by squaring target RMSE per seed, then averaging target MSE values within seed and aggregating across seeds.

## Main MSE Summary

| Dataset | Best overall | Best raw tree | PRISM | Simple MLP | Two-stage MLP | CNN-MLP | FT-Transformer | Oracle MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `airfoil_self_noise` | `prism` 2.2979 | `random_forest` 4.5540 | 2.2979 | 9.4427 | 6.8922 | 2.7012 | 4.0997 | 5.6777 |
| `energy_efficiency` | `prism` 0.9836 | `xgboost` 1.4183 | 0.9836 | 1.8255 | 3.0073 | 2.0889 | 6.6087 | 1.6293 |
| `hydraulic_systems` | `oracle_mlp` 4.3224 | `xgboost` 7.0868 | 6.9201 | 8.5325 | 7.6372 | 15.3255 | 14.3293 | 4.3224 |
| `household_power_consumption` | `oracle_mlp` 0.0028 | `lightgbm` 0.0033 | 0.0031 | 0.0036 | 0.0037 | 0.0037 | 0.0039 | 0.0028 |
| `concrete_slump_test` | `prism` 6.5184 | `xgboost` 14.3609 | 6.5184 | 6.6413 | 7.3481 | 22.0125 | 22.5135 | 9.1535 |

## Case1 MSE Summary

| Dataset | Best overall | Best raw tree | PRISM | Simple MLP | Two-stage MLP | CNN-MLP | FT-Transformer | Oracle MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wine_quality` | `prism_z_acp_random_forest` 0.4407 | `random_forest` 0.4454 | 0.5389 | 0.5503 | 0.5454 | 0.5717 | 0.5860 | 0.4814 |
| `superconductivity` | `random_forest` 94.3372 | `random_forest` 94.3372 | 110.7779 | 104.5519 | 105.4407 | 133.7914 | 115.3266 | 100.2824 |
| `auto_mpg` | `random_forest` 8.8242 | `random_forest` 8.8242 | 10.5637 | 10.5732 | 10.4325 | 9.9202 | 10.4637 | 9.8029 |
| `combined_cycle_power_plant` | `random_forest` 12.8155 | `random_forest` 12.8155 | 15.7984 | 16.8363 | 16.7315 | 16.4887 | 16.7959 | 14.7303 |

## Case2 MSE Summary

| Dataset | Best overall | Best raw tree | PRISM | Simple MLP | Two-stage MLP | CNN-MLP | FT-Transformer | Oracle MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `air_quality` | `oracle_mlp` 2597.0011 | `random_forest` 12412.7415 | 18054.0747 | 17278.6057 | 17197.5145 | 21469.8143 | 16095.9494 | 2597.0011 |
| `parkinsons_telemonitoring` | `oracle_mlp` 40.9529 | `xgboost` 171.0110 | 175.6581 | 200.7256 | 232.3202 | 144.7270 | 170.9682 | 40.9529 |
| `real_estate_valuation` | `oracle_mlp` 72.5991 | `lightgbm` 163.1063 | 179.8903 | 171.6847 | 173.7356 | 170.0201 | 169.0996 | 72.5991 |
| `student_performance` | `oracle_mlp` 3.3689 | `xgboost` 9.9581 | 11.2234 | 11.2828 | 11.2092 | 13.6827 | 13.7548 | 3.3689 |

## PRISM vs Deep X-only Backbones

| Group | Dataset | PRISM MSE | CNN-MLP MSE | FT-Transformer MSE | PRISM vs CNN | PRISM vs FT |
|---|---|---:|---:|---:|---:|---:|
| Main | `airfoil_self_noise` | 2.2979 | 2.7012 | 4.0997 | 14.9% | 43.9% |
| Main | `energy_efficiency` | 0.9836 | 2.0889 | 6.6087 | 52.9% | 85.1% |
| Main | `hydraulic_systems` | 6.9201 | 15.3255 | 14.3293 | 54.8% | 51.7% |
| Main | `household_power_consumption` | 0.0031 | 0.0037 | 0.0039 | 17.7% | 21.5% |
| Main | `concrete_slump_test` | 6.5184 | 22.0125 | 22.5135 | 70.4% | 71.0% |
| Case1 | `wine_quality` | 0.5389 | 0.5717 | 0.5860 | 5.7% | 8.0% |
| Case1 | `superconductivity` | 110.7779 | 133.7914 | 115.3266 | 17.2% | 3.9% |
| Case1 | `auto_mpg` | 10.5637 | 9.9202 | 10.4637 | -6.5% | -1.0% |
| Case1 | `combined_cycle_power_plant` | 15.7984 | 16.4887 | 16.7959 | 4.2% | 5.9% |
| Case2 | `air_quality` | 18054.0747 | 21469.8143 | 16095.9494 | 15.9% | -12.2% |
| Case2 | `parkinsons_telemonitoring` | 175.6581 | 144.7270 | 170.9682 | -21.4% | -2.7% |
| Case2 | `real_estate_valuation` | 179.8903 | 170.0201 | 169.0996 | -5.8% | -6.4% |
| Case2 | `student_performance` | 11.2234 | 13.6827 | 13.7548 | 18.0% | 18.4% |
