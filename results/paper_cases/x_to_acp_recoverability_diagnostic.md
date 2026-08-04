# X-to-ACP Recoverability Diagnostic

This diagnostic measures whether deployable inputs X contain enough information to reconstruct privileged ACP variables. It is separate from target prediction: models are trained as X -> ACP and evaluated on the ACP test set using the same dataset preprocessing and split policy as `run_processed_data_experiments.py`.

- Primary recoverability model in the tables below: `random_forest`.
- `macro R2`: mean ACP-descriptor R2 across ACP dimensions; higher is better.
- `standardized RMSE`: mean ACP RMSE after train-only ACP standardization; lower is better.
- `Oracle gain`: relative RMSE reduction of `oracle_mlp` over `simple_mlp` from the target-prediction experiment.
- Case2 evidence: high Oracle gain plus low X->ACP macro R2 indicates that ACP is target-relevant but difficult to recover from feasible inputs.
- This diagnostic should not be used as the only grouping rule. Main grouping also depends on observed feasible PRISM gains over baselines, while Case1 is primarily defined by small oracle gain.

## Main

| Dataset | X->ACP macro R2 | X->ACP standardized RMSE | Oracle gain | Interpretation |
|---|---:|---:|---:|---|
| `airfoil_self_noise` | -1.0000 +/- 0.0001 | 1.2987 +/- 0.0768 | 22.5% | Main-group target result; recoverability is diagnostic, not the sole grouping rule. |
| `energy_efficiency` | -0.9966 +/- 0.0071 | 1.3627 +/- 0.0230 | 5.4% | Main-group target result; recoverability is diagnostic, not the sole grouping rule. |
| `hydraulic_systems` | 0.8809 +/- 0.0144 | 0.2019 +/- 0.0215 | 23.9% | High ACP recoverability and meaningful oracle gain support the main success pattern. |
| `household_power_consumption` | 0.4826 +/- 0.0012 | 0.6464 +/- 0.0022 | 11.4% | Moderate ACP recoverability and feasible PRISM gain support the main success pattern. |
| `concrete_slump_test` | 0.3469 +/- 0.2364 | 0.8100 +/- 0.1795 | -13.5% | Small-data exception; use as a secondary/appendix result rather than core recoverability evidence. |

## Case1

| Dataset | X->ACP macro R2 | X->ACP standardized RMSE | Oracle gain | Interpretation |
|---|---:|---:|---:|---|
| `wine_quality` | 0.7137 +/- 0.0041 | 0.5181 +/- 0.0089 | 6.5% | Oracle gain is small; ACP adds limited target-relevant information. |
| `superconductivity` | 0.9316 +/- 0.0030 | 0.2538 +/- 0.0069 | 2.1% | Oracle gain is small; ACP adds limited target-relevant information. |
| `auto_mpg` | 0.7192 +/- 0.0496 | 0.5009 +/- 0.0439 | 4.0% | Oracle gain is small; ACP adds limited target-relevant information. |
| `combined_cycle_power_plant` | -0.6925 +/- 0.0131 | 1.2607 +/- 0.0298 | 6.5% | Oracle gain is small; ACP adds limited target-relevant information. |

## Case2

| Dataset | X->ACP macro R2 | X->ACP standardized RMSE | Oracle gain | Interpretation |
|---|---:|---:|---:|---|
| `air_quality` | 0.0024 +/- 0.0095 | 0.9761 +/- 0.0016 | 59.1% | High oracle gain with limited ACP recoverability; supports Case2. |
| `parkinsons_telemonitoring` | -0.8930 +/- 0.0917 | 1.3666 +/- 0.1351 | 55.1% | High oracle gain with limited ACP recoverability; supports Case2. |
| `real_estate_valuation` | 0.0537 +/- 0.1377 | 0.9627 +/- 0.1272 | 35.6% | High oracle gain with limited ACP recoverability; supports Case2. |
| `student_performance` | 0.2781 +/- 0.0938 | 0.8280 +/- 0.0682 | 45.4% | High oracle gain with limited ACP recoverability; supports Case2. |
