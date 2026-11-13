from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COLS = ["UnderwrittenCoverID","PolicyID","TransactionMonth","Gender","Province","PostalCode","VehicleType","RegistrationYear","make","Model","TotalPremium","TotalClaims"]

def load_data(path):
    return pd.read_csv(path, sep="|", usecols=COLS, low_memory=False)

def aggregate_policies(rows):
    p = rows.groupby("PolicyID", dropna=False).agg(
        province=("Province","first"), postal_code=("PostalCode","first"),
        gender=("Gender","first"), premium=("TotalPremium","sum"),
        claims=("TotalClaims","sum"), row_count=("PolicyID","size"),
        vehicle_type=("VehicleType","first"), registration_year=("RegistrationYear","first"),
        make=("make","first"), model=("Model","first")).reset_index()
    for c in ["province","gender","vehicle_type","make","model"]:
        p[c] = p[c].fillna("Unknown").astype(str).str.strip().replace("","Unknown")
    p["postal_code"] = p["postal_code"].fillna(-1).astype(int).astype(str)
    p["registration_year"] = pd.to_numeric(p["registration_year"], errors="coerce")
    p["has_claim"] = (p.claims > 0).astype(int)
    p["margin_proxy"] = p.premium - p.claims
    return p

def profile_all_columns(path, chunksize=50000):
    total, missing, dtypes = 0, None, {}
    for chunk in pd.read_csv(path, sep="|", chunksize=chunksize, low_memory=False):
        total += len(chunk)
        m = chunk.isna().sum()
        missing = m if missing is None else missing.add(m, fill_value=0)
        for c, dtype in chunk.dtypes.items():
            dtypes.setdefault(c,set()).add(str(dtype))
    profile = pd.DataFrame({"missing_count":missing.astype(int),
        "missing_pct":(missing/total*100).round(3),
        "dtype":{c:" / ".join(sorted(v)) for c,v in dtypes.items()}})
    return profile.sort_values("missing_pct",ascending=False)

def hypothesis_tests(p, min_zip_policies=30):
    out = {}
    prov = pd.crosstab(p.province, p.has_claim)
    q = stats.chi2_contingency(prov)
    out["province_claim_frequency"] = {"statistic":float(q.statistic),"p_value":float(q.pvalue),"groups":len(prov)}
    sizes = p.groupby("postal_code").size()
    eligible = sizes[sizes >= min_zip_policies].index
    z = p[p.postal_code.isin(eligible)]
    q = stats.chi2_contingency(pd.crosstab(z.postal_code,z.has_claim))
    out["postal_code_claim_frequency"] = {"statistic":float(q.statistic),"p_value":float(q.pvalue),"groups":len(eligible),"minimum_n":min_zip_policies}
    q = stats.kruskal(*[g.margin_proxy.dropna().values for _,g in z.groupby("postal_code")])
    out["postal_code_margin_proxy"] = {"statistic":float(q.statistic),"p_value":float(q.pvalue),"groups":len(eligible)}
    g = p[p.gender.str.lower().isin(["male","female"])].copy()
    tab = pd.crosstab(g.gender,g.has_claim)
    if {"Male","Female"}.issubset(set(tab.index)):
        q = stats.chi2_contingency(tab.loc[["Female","Male"]])
        out["gender_claim_frequency"] = {"statistic":float(q.statistic),"p_value":float(q.pvalue),"n":len(g)}
        q = stats.ttest_ind(g.loc[g.gender=="Female","claims"],g.loc[g.gender=="Male","claims"],equal_var=False,nan_policy="omit")
        out["gender_claim_amount_welch"] = {"statistic":float(q.statistic),"p_value":float(q.pvalue),"n_female":int((g.gender=="Female").sum()),"n_male":int((g.gender=="Male").sum())}
    provinces = p.groupby("province").agg(policies=("PolicyID","nunique"),claim_rate=("has_claim","mean"),avg_claims=("claims","mean"),avg_premium=("premium","mean"),avg_margin_proxy=("margin_proxy","mean")).reset_index().sort_values("claim_rate",ascending=False)
    postal = p.groupby("postal_code").agg(policies=("PolicyID","nunique"),claim_rate=("has_claim","mean"),avg_claims=("claims","mean"),avg_premium=("premium","mean"),avg_margin_proxy=("margin_proxy","mean")).reset_index()
    postal = postal[postal.policies>=min_zip_policies].sort_values("claim_rate",ascending=False)
    return out, provinces, postal

def train_models(p, random_state=42):
    features = ["province","postal_code","gender","vehicle_type","registration_year","make","model"]
    X = p[features].copy()
    cat = [c for c in features if c!="registration_year"]
    def prep():
        return ColumnTransformer([("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",min_frequency=10))]),cat),
                                  ("num",Pipeline([("impute",SimpleImputer(strategy="median"))]),["registration_year"])])
    Xtr,Xte,ytr,yte = train_test_split(X,p.claims,test_size=.2,random_state=random_state)
    reg = Pipeline([("prep",prep()),("model",RandomForestRegressor(n_estimators=150,min_samples_leaf=3,n_jobs=-1,random_state=random_state))])
    reg.fit(Xtr,ytr); pred=reg.predict(Xte)
    names=reg.named_steps["prep"].get_feature_names_out()
    grouped={}
    for name,imp in zip(names,reg.named_steps["model"].feature_importances_):
        raw=name.split("__",1)[-1]
        field=next((c for c in sorted(features,key=len,reverse=True) if raw==c or raw.startswith(c+"_")),raw)
        grouped[field]=grouped.get(field,0)+float(imp)
    regression={"target":"policy-level TotalClaims sum","test_size":len(yte),"MAE":float(mean_absolute_error(yte,pred)),"RMSE":float(mean_squared_error(yte,pred)**.5),"R2":float(r2_score(yte,pred)),"grouped_feature_importance":dict(sorted(grouped.items(),key=lambda x:x[1],reverse=True))}
    Xtr,Xte,ytr,yte=train_test_split(X,p.has_claim,test_size=.2,random_state=random_state,stratify=p.has_claim)
    clf=Pipeline([("prep",prep()),("model",RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight="balanced",n_jobs=-1,random_state=random_state))])
    clf.fit(Xtr,ytr); pred=clf.predict(Xte); prob=clf.predict_proba(Xte)[:,1]
    classification={"target":"any positive policy-level TotalClaims","test_size":len(yte),"precision":float(precision_score(yte,pred,zero_division=0)),"recall":float(recall_score(yte,pred,zero_division=0)),"F1":float(f1_score(yte,pred,zero_division=0)),"ROC_AUC":float(roc_auc_score(yte,prob))}
    return {"regression":regression,"classification":classification}

def run_analysis(path, output="reports/generated"):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    rows=load_data(path); p=aggregate_policies(rows)
    dt=pd.to_datetime(rows.TransactionMonth,errors="coerce")
    desc={"source_rows":len(rows),"columns":52,"unique_policies":int(rows.PolicyID.nunique()),"unique_cover_ids":int(rows.UnderwrittenCoverID.nunique()),"date_min":str(dt.min()),"date_max":str(dt.max()),"policy_claim_rate":float(p.has_claim.mean()),"total_premium":float(rows.TotalPremium.sum()),"total_claims":float(rows.TotalClaims.sum()),"negative_premium_rows":int((rows.TotalPremium<0).sum()),"negative_claim_rows":int((rows.TotalClaims<0).sum())}
    profile=profile_all_columns(path)
    tests,provinces,postal=hypothesis_tests(p)
    models=train_models(p)
    for name,obj in [("descriptive_summary.json",desc),("hypothesis_tests.json",tests),("model_metrics.json",models)]:
        (out/name).write_text(json.dumps(obj,indent=2))
    profile.to_csv(out/"full_source_data_quality.csv")
    provinces.to_csv(out/"province_summary.csv",index=False)
    postal.to_csv(out/"postal_code_summary_min30.csv",index=False)
    return {"descriptive":desc,"tests":tests,"models":models,"province_summary":provinces,"postal_summary":postal}
