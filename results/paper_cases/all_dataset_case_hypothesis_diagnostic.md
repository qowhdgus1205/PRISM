# All-Dataset Case Hypothesis Diagnostic

This table applies the same diagnostic logic to every external benchmark dataset, rather than assuming the current Main/Case1/Case2 grouping is always pure. It separates two questions: whether ACP is target-relevant, measured by oracle gain, and whether ACP is recoverable from deployable X, measured by X-to-ACP macro R2 and standardized RMSE.

Diagnostic thresholds used only as interpretive flags: Case1-like if oracle gain < 10%; Case2-like if oracle gain >= 30% and X-to-ACP macro R2 < 0.30 or standardized RMSE > 0.80; Main-like if oracle gain >= 10%, macro R2 >= 0.30, and PRISM improves over Simple MLP.

| Current group | Dataset | Oracle gain | X->ACP R2 | X->ACP std RMSE | PRISM vs Simple | PRISM vs Two-stage | Diagnostic interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Main | `airfoil_self_noise` | 22.5% | -1.0000 | 1.2987 | 50.7% | 42.2% | Mixed: ACP may help, but recoverability is weak; PRISM success must be argued empirically. |
| Main | `concrete_slump_test` | -13.5% | 0.3469 | 0.8100 | -0.2% | 4.2% | Case1-like: oracle gain is small, so ACP adds limited target-relevant information. |
| Main | `energy_efficiency` | 5.4% | -0.9966 | 1.3627 | 26.8% | 39.3% | Case1-like: oracle gain is small, so ACP adds limited target-relevant information. |
| Main | `household_power_consumption` | 11.4% | 0.4826 | 0.6464 | 7.6% | 9.2% | Main-like: ACP is useful, recoverable enough, and PRISM improves over Simple MLP. |
| Main | `hydraulic_systems` | 23.9% | 0.8809 | 0.2019 | 9.9% | 4.7% | Main-like: ACP is useful, recoverable enough, and PRISM improves over Simple MLP. |
| Case1 | `auto_mpg` | 4.0% | 0.7192 | 0.5009 | 0.4% | -0.5% | Case1-like: oracle gain is small, so ACP adds limited target-relevant information. |
| Case1 | `combined_cycle_power_plant` | 6.5% | -0.6925 | 1.2607 | 3.1% | 2.8% | Case1-like: oracle gain is small, so ACP adds limited target-relevant information. |
| Case1 | `superconductivity` | 2.1% | 0.9316 | 0.2538 | -2.5% | -2.1% | Case1-like: oracle gain is small, so ACP adds limited target-relevant information. |
| Case1 | `wine_quality` | 6.5% | 0.7137 | 0.5181 | 1.0% | 0.6% | Case1-like: oracle gain is small, so ACP adds limited target-relevant information. |
| Case2 | `air_quality` | 59.1% | 0.0024 | 0.9761 | -2.0% | -2.3% | Case2-like: oracle gain is large but X-to-ACP recoverability is weak. |
| Case2 | `parkinsons_telemonitoring` | 55.1% | -0.8930 | 1.3666 | 6.6% | 12.1% | Case2-like: oracle gain is large but X-to-ACP recoverability is weak. |
| Case2 | `real_estate_valuation` | 35.6% | 0.0537 | 0.9627 | -2.3% | -1.6% | Case2-like: oracle gain is large but X-to-ACP recoverability is weak. |
| Case2 | `student_performance` | 45.4% | 0.2781 | 0.8280 | 0.4% | -0.2% | Case2-like: oracle gain is large but X-to-ACP recoverability is weak. |

## Implications

- Current Main contains mixed cases. `hydraulic_systems` and `household_power_consumption` match the intended recoverable-ACP pattern most directly. `airfoil_self_noise` and `energy_efficiency` have weak RF X-to-ACP recoverability but PRISM still performs well, so they should be argued as empirical PRISM success cases rather than clean recoverability examples. `concrete_slump_test` is small and has negative oracle gain, so it is better treated cautiously or as an appendix/small-data case.
- Current Case1 is coherent overall: oracle gains are below 10%, so ACP target relevance is limited. Some datasets still have high X-to-ACP R2, which reinforces the point that recoverability alone is not enough when ACP does not add much target information.
- Current Case2 is coherent: all four datasets have high oracle gains and low X-to-ACP recoverability, directly supporting the transfer-limitation explanation.
