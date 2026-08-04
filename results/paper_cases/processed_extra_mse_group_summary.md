## External Results in MSE

MSE is derived from the seed-level target RMSE values in `processed_extra_results.csv`: each target RMSE is squared per seed, target MSE values are averaged within seed, and then mean/min/max/std are aggregated across the 5 seeds. This avoids using `(mean RMSE)^2`, which is not the same statistic.

- Seed-level MSE file: `results/uci_external_experiments/processed_extra_mse_results.csv`
- Aggregate MSE file: `results/uci_external_experiments/processed_extra_mse_aggregate_with_minmax.csv`

### Main MSE Summary

| Dataset | Best overall | Best tree | PRISM | Simple MLP | Two-stage MLP | Oracle MLP |
|---|---:|---:|---:|---:|---:|---:|
| `airfoil_self_noise` | `prism` 2.2024 | `random_forest` 4.5540 | 2.2024 | 9.4427 | 6.3972 | 5.4572 |
| `energy_efficiency` | `prism` 1.0067 | `xgboost` 1.4183 | 1.0067 | 1.8255 | 2.9638 | 1.6714 |
| `hydraulic_systems` | `oracle_mlp` 4.0238 | `xgboost` 7.0868 | 6.0363 | 8.5325 | 7.2047 | 4.0238 |
| `household_power_consumption` | `oracle_mlp` 0.0028 | `lightgbm` 0.0033 | 0.0031 | 0.0036 | 0.0038 | 0.0028 |
| `concrete_slump_test` | `simple_mlp` 6.6413 | `xgboost` 14.3609 | 6.9058 | 6.6413 | 8.2972 | 12.0214 |

### Case1 MSE Summary

| Dataset | Best overall | Best tree | PRISM | Simple MLP | Two-stage MLP | Oracle MLP |
|---|---:|---:|---:|---:|---:|---:|
| `wine_quality` | `prism_z_random_forest` 0.4439 | `random_forest` 0.4454 | 0.5519 | 0.5503 | 0.5549 | 0.4801 |
| `superconductivity` | `random_forest` 94.3372 | `random_forest` 94.3372 | 111.9001 | 104.5519 | 106.0020 | 99.4810 |
| `auto_mpg` | `random_forest` 8.8242 | `random_forest` 8.8242 | 10.4480 | 10.5732 | 10.0973 | 9.3082 |
| `combined_cycle_power_plant` | `random_forest` 12.8155 | `random_forest` 12.8155 | 15.9077 | 16.8363 | 16.7236 | 14.8304 |

### Case2 MSE Summary

| Dataset | Best overall | Best tree | PRISM | Simple MLP | Two-stage MLP | Oracle MLP |
|---|---:|---:|---:|---:|---:|---:|
| `air_quality` | `oracle_mlp` 2532.7757 | `random_forest` 12412.7415 | 17377.5290 | 17278.6057 | 17069.4852 | 2532.7757 |
| `parkinsons_telemonitoring` | `oracle_mlp` 45.7675 | `xgboost` 171.0110 | 181.5278 | 200.7256 | 206.8690 | 45.7675 |
| `real_estate_valuation` | `oracle_mlp` 71.0631 | `lightgbm` 163.1063 | 183.4229 | 171.6847 | 180.5625 | 71.0631 |
| `student_performance` | `oracle_mlp` 3.2675 | `xgboost` 9.9581 | 11.1919 | 11.2828 | 11.3097 | 3.2675 |
