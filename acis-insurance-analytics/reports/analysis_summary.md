# ACIS analysis summary

- Source rows: **1,000,098**; source columns: **52**
- Distinct policies: **7,000**; underwritten cover IDs: **116,532**
- Transaction range: **2013-10-01 to 2015-08-01**
- Policy claim frequency: **20.3%**
- TotalPremium sum: **61,911,562.70**; TotalClaims sum: **64,867,546.17**
- Negative premium rows: **288**; negative claims rows: **5**

## Statistical tests (alpha 0.05)

| Hypothesis | Test | p-value | Decision |
|---|---|---:|---|
| Province claim frequency | Chi-square | 1.91e-05 | Reject equal rates |
| Postal-code claim frequency (42 groups, >=30 policies) | Chi-square | 1.08e-08 | Reject equal rates |
| Postal-code margin proxy (42 groups, >=30 policies) | Kruskal–Wallis | 0.00113 | Reject equal distributions |
| Female/Male claim frequency (n=267) | Chi-square | 0.440 | Do not reject |
| Female/Male claim amount (n=35 vs 232) | Welch t-test | 0.772 | Do not reject |

## Regression comparison (random policy-level 80/20 holdout)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 15,677 | 35,406 | 0.016 |
| Decision Tree | 15,100 | 36,107 | -0.023 |
| Random Forest | 14,741 | 36,490 | -0.045 |
| XGBoost | 14,914 | 35,411 | 0.016 |

Random Forest classifier for any claim: precision 0.293, recall 0.577, F1 0.389, ROC-AUC 0.651.

Postal-code Linear Regression (42 groups with >=30 policies): median holdout R² 0.000; 19 groups had positive R². SHAP mean absolute XGBoost attributions across 100 holdout policies ranked Model, PostalCode, Province, RegistrationYear, make, Gender and VehicleType.

## Full-field data quality

Most missing: NumberOfVehiclesInFleet 100%; CrossBorder 99.93%; CustomValueEstimate 77.96%; Rebuilt/Converted/WrittenOff 64.18% each; NewVehicle 15.33%; Bank 14.60%.

The regression models have negligible explanatory power; the classifier is weak. Premium minus claims is only a margin proxy, not underwriting profit. No optimal-premium label is available. Do not use these baselines for pricing or underwriting.
