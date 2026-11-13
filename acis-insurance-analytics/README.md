# ACIS Motor Insurance Analytics

A reproducible analysis project for the AlphaCare Insurance Solutions (ACIS) motor-insurance challenge.

## Dataset

Source: `MachineLearningRating_v3.txt`, pipe-delimited, 529,363,713 bytes. The supplied DVC pointer hash and byte size were verified against the uploaded file. The source has 1,000,098 rows, 52 columns, 116,532 underwritten-cover IDs and 7,000 policies. It spans October 2013–August 2015 (the challenge brief describes February 2014–August 2015; this analysis follows the actual file).

The 529 MB source data is deliberately excluded from GitHub. Place it at `data/raw/MachineLearningRating_v3.txt`. Its pointer is in `data/raw/MachineLearningRating_v3.txt.dvc`. The pointer alone cannot restore the dataset: configure a private DVC remote before using `dvc pull`.

## Run

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_analysis.py --data data/raw/MachineLearningRating_v3.txt
pytest -q
streamlit run app.py
```

The pipeline profiles all 52 source columns in chunks, aggregates monthly cover rows to policy level, runs hypothesis tests, and trains policy-level baseline models.

## Results from the supplied file

- 7,000 policies; 20.3% have positive policy-level aggregated claims.
- Province claim-frequency chi-square: p = 1.91e-05 (reject equal rates).
- Postal-code claim-frequency chi-square: p = 1.08e-08 across 42 groups with at least 30 policies.
- Postal-code margin-proxy Kruskal–Wallis: p = 0.00113 across the same 42 groups.
- Recorded Male/Female claim-frequency chi-square: p = 0.440; only 267 policies have gender explicitly recorded (232 Male, 35 Female).
- Recorded Male/Female claim-amount Welch t-test: p = 0.772.

Baseline model diagnostics on a random 80/20 policy-level split:
- Claims regression: MAE 14,741; RMSE 36,490; R² -0.045.
- Any-claim classification: precision 0.293; recall 0.577; F1 0.389; ROC-AUC 0.651.

The negative regression R² means this baseline is worse than predicting the test-set mean. The classifier is weak and is not suitable for underwriting decisions.

## Important caveats

- The aggregation sums monthly `TotalPremium` and `TotalClaims` by PolicyID. Claim frequency is the proportion of policies with positive aggregated claims.
- `TotalPremium - TotalClaims` is a margin proxy, not true underwriting profit; expenses, commissions, reserves, reinsurance and capital costs are absent.
- No observed optimal-premium label or defined pricing objective is supplied. The project does not claim to predict an optimal premium.
- Gender data is mostly unspecified. Do not infer gender or use these exploratory tests to justify discriminatory pricing.
- Results are observational and unadjusted for confounding. Statistical significance is not causation.
- Regression must be evaluated with MAE/RMSE/R²; precision/recall/F1 are classification metrics.

## Deployment

The Streamlit app and Dockerfile are included. Hosting requires connecting this branch in Streamlit Community Cloud (or another host) and ensuring the dataset is available privately at runtime. Keep the app private; do not commit the raw data or expose it through a public dashboard. Hosting is not activated by this repository commit.
