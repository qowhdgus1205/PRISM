# Paper datasets

The public PRISM package uses the following 13 processed datasets. Raw UCI
archives are not redistributed. All processed tables are below the project's
100,000-row experiment limit.

| Group | Dataset | Rows | UCI ID / DOI |
|---|---|---:|---|
| Main | Airfoil Self-Noise | 1,503 | [291 / 10.24432/C5VW2C](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) |
| Main | Energy Efficiency | 768 | [242 / 10.24432/C51307](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| Main | Condition Monitoring of Hydraulic Systems | 2,205 | [447 / 10.24432/C5CW21](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| Main | Individual Household Electric Power Consumption | 34,168 | [235 / 10.24432/C58K54](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) |
| Main | Concrete Slump Test | 103 | [182 / 10.24432/C5FG7D](https://archive.ics.uci.edu/dataset/182/concrete+slump+test) |
| Case1 | Wine Quality | 6,497 | [186 / 10.24432/C56S3T](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Case1 | Superconductivity | 21,263 | [464 / 10.24432/C53P47](https://archive.ics.uci.edu/dataset/464/superconductivty+data) |
| Case1 | Auto MPG | 398 | [9 / 10.24432/C5859H](https://archive.ics.uci.edu/dataset/9/auto+mpg) |
| Case1 | Combined Cycle Power Plant | 9,568 | [294 / 10.24432/C5002N](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) |
| Case2 | Air Quality | 9,357 | [360 / 10.24432/C59K5F](https://archive.ics.uci.edu/dataset/360/air+quality) |
| Case2 | Parkinsons Telemonitoring | 5,875 | [189 / 10.24432/C5ZS3N](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) |
| Case2 | Real Estate Valuation | 414 | [477 / 10.24432/C5J30W](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) |
| Case2 | Student Performance | 1,044 | [320 / 10.24432/C5TG7T](https://archive.ics.uci.edu/dataset/320/student+performance) |

The row counts above refer to the PRISM-processed CSVs, not the upstream raw
files. The preparation code documents the exact transformation from raw fields
to `feature_*`, `intermediate_*`, `target_*`, and `ID_*` roles.

The UCI pages linked above identify these datasets as CC BY 4.0. Redistribution
of transformed data requires attribution to the original creators; retain this
file with any redistributed copy. PRISM source code is distributed separately
under the root MIT `LICENSE`.
