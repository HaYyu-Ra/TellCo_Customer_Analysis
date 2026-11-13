# TellCo Telecom Analytics

Streamlit dashboard for the 10 Academy TellCo xDR challenge: user overview, engagement, network experience and satisfaction-proxy analysis.

## Deploy
This repository is prepared for Streamlit Community Cloud:
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Create app from `HaYyu-Ra/TellCo_Customer_Analysis`.
3. Branch: `deploy/tellco-analytics`; main file: `app.py`; Python: 3.11.
4. Deploy. The app can start without a data source and will display setup instructions.
5. For real analysis, upload a CSV/Parquet/Excel xDR extract in the sidebar or configure a secure `TELLCO_DATABASE_URL` in Streamlit Secrets. Never commit credentials or raw subscriber records.


## Private online access
After the first deployment on Streamlit Community Cloud:
1. Open the app's **Share** menu (or **Manage app → Settings → Sharing**).
2. Under **Who can view this app**, choose **Only specific people can view this app** and save.
3. Add each approved viewer by email in the app's **Share** menu. Viewers sign in with Google or a one-time email link.
4. Keep the xDR data out of GitHub. A sidebar file upload is session-scoped; each viewer/session may need to upload the file again. For shared, persistent data access, provision a hosted PostgreSQL database and add its connection string as the `TELLCO_DATABASE_URL` secret in Streamlit app settings. Never commit database credentials.

The app is not hosted until an authorized repository owner completes the initial Streamlit sign-in and deployment. The repository code alone does not create a live app.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## PostgreSQL restore
The supplied telecom database is a PostgreSQL custom-format archive (PGDMP), not a plain SQL script. Use `pg_restore`, not `psql -f`:
```bash
pg_restore --list telecom.sql
createdb tellco_analysis
pg_restore --no-owner --no-privileges -d tellco_analysis telecom.sql
```
Set `TELLCO_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/tellco_analysis` in the hosting secrets.

## Analysis scope
Device/manufacturer rankings; per-customer sessions, duration and traffic; application totals; engagement and experience K-means; satisfaction proxy; PCA; data-quality diagnostics and downloadable scores.

## Important limitations
Challenge-defined satisfaction scores are Euclidean-distance proxies, not survey results. Regression is an in-sample diagnostic where the target is derived from the same telemetry. A one-month xDR sample does not establish profitability or investment value. Financial statements, revenue, cost, churn, capex, spectrum obligations, debt and valuation are required for a purchase decision.

## CI and container
```bash
pytest -q
docker build -t tellco-analytics .
docker run --rm -p 8501:8501 tellco-analytics
```
