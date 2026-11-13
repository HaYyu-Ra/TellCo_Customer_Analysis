# ACIS analysis results

- Source rows: **1,000,098**; source columns: **52**
- Distinct policies: **7,000**; underwritten cover IDs: **116,532**
- Transaction range: **2013-10-01 to 2015-08-01**
- Policy claim frequency: **20.3%**
- Premium sum: **61,911,562.70**; claims sum: **64,867,546.17**

| Test | p-value | Decision |
|---|---:|---|
| Province claim frequency (chi-square) | 1.91e-05 | Reject equal rates |
| Postal-code claim frequency (42 groups, n>=30; chi-square) | 1.08e-08 | Reject equal rates |
| Postal-code margin proxy (42 groups, n>=30; Kruskal-Wallis) | 0.00113 | Reject equal distributions |
| Female/Male claim frequency (chi-square, n=267) | 0.440 | Do not reject |
| Female/Male claim amount (Welch t, n=35 vs 232) | 0.772 | Do not reject |

Model holdout (random 80/20 policy split):
- Claims regression: MAE 14,741; RMSE 36,490; R² -0.045.
- Any-claim classifier: precision 0.293; recall 0.577; F1 0.389; ROC-AUC 0.651.

Missingness leaders: NumberOfVehiclesInFleet 100%; CrossBorder 99.93%; CustomValueEstimate 77.96%; WrittenOff/Rebuilt/Converted 64.18%.

The negative regression R² and weak classifier mean these baselines are not fit for pricing decisions. Premium less claims is a margin proxy, not underwriting profit. The source has no optimal-premium target.
