# AlphaCare Insurance Solutions (ACIS)

## Motor Insurance Analytics

A reproducible analysis project for the AlphaCare Insurance Solutions (ACIS) motor-insurance challenge.

## Dataset and privacy

The source is `MachineLearningRating_v3.txt`, pipe-delimited, 529,363,713 bytes. It has 1,000,098 rows, 52 columns, 116,532 underwritten-cover IDs and 7,000 policies. It spans October 2013–August 2015; the brief states February 2014–August 2015, so analysis follows the actual file.

The 529 MB source data is excluded from GitHub. Put it at `data/raw/MachineLearningRating_v3.txt`. The DVC pointer is `data/raw/MachineLearningRating_v3.txt.dvc`; its MD5 and byte size were verified against the uploaded file. A pointer alone cannot restore the data.

For private DVC tracking:

```bash
pip install -r requirements.txt
dvc init
dvc remote add -d storage /path/to/private/dvc-storage
dvc add data/raw/MachineLearningRating_v3.txt
git add .dvc data/raw/MachineLearningRating_v3.txt.dvc .gitignore
git commit -m "Track ACIS source data with DVC"
dvc push
```

Use a private cloud remote and credentials outside Git for team access. No DVC remote credentials are committed.

## Run locally

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_analysis.py --data data/raw/MachineLearningRating_v3.txt
pytest -q
streamlit run app.py
```

The pipeline profiles all 52 fields in 50,000-row chunks, aggregates monthly cover rows to policy level, runs hypothesis tests, and compares Linear Regression, Decision Tree, Random Forest and XGBoost models. It also runs separate Linear Regression models for sufficiently populated postal codes and calculates mean absolute SHAP values for XGBoost.

## Findings from the uploaded dataset

- 7,000 policies; 20.3% have positive policy-level aggregated claims.
- Province claim-frequency chi-square: p = 1.91e-05.
- Postal-code claim-frequency chi-square: p = 1.08e-08 across 42 groups with at least 30 policies.
- Postal-code margin-proxy Kruskal–Wallis: p = 0.00113 across the same 42 groups.
- Recorded Male/Female claim-frequency chi-square: p = 0.440; only 267 policies have gender explicitly recorded (232 Male, 35 Female).
- Recorded Male/Female claim-amount Welch t-test: p = 0.772.

Regression holdout (random 80/20 policy-level split):

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 15,677 | 35,406 | 0.016 |
| Decision Tree | 15,100 | 36,107 | -0.023 |
| Random Forest | 14,741 | 36,490 | -0.045 |
| XGBoost | 14,914 | 35,411 | 0.016 |

Linear Regression has the highest R², but 0.016 is negligible explanatory power. The Random Forest claim classifier has precision 0.293, recall 0.577, F1 0.389 and ROC-AUC 0.651.

Postal-code Linear Regression was evaluated for 42 postal codes with at least 30 policies. Median holdout R² was 0.000; 19 of 42 groups had positive R². The small within-group test sets make estimates unstable.

Mean absolute SHAP attributions for XGBoost over 100 holdout policies ranked Model, PostalCode, Province, RegistrationYear, make, Gender and VehicleType. SHAP attributions are not causal effects.

## Caveats

- Monthly cover rows are summed by PolicyID. Claim frequency is the share of policies with positive aggregated claims.
- `TotalPremium - TotalClaims` is only a margin proxy, not true underwriting profit; expenses, commissions, reserves, reinsurance and capital costs are absent.
- There is no observed optimal-premium label or defined pricing objective. The project does not predict an optimal premium.
- Gender is mostly unspecified. Do not infer gender or use exploratory tests to justify discriminatory pricing.
- Results are observational and unadjusted for confounding. Statistical significance is not causation.
- Regression uses MAE/RMSE/R²; precision/recall/F1 are classification metrics.

## Deployment

In Streamlit Community Cloud choose **New app**, repository `HaYyu-Ra/TellCo_Customer_Analysis`, branch `acis-insurance-analysis`, main file `acis-insurance-analytics/app.py`. Set visibility to private. The app has a 600 MB upload limit, but a 529 MB upload may exceed low-memory hosting limits; a private DVC/object-storage remote is preferable. The repository commit does not activate hosting: the account owner must authorize the Streamlit connection and provide private runtime data access.
