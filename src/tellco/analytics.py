"""Compact, reusable TellCo xDR analytics pipeline."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

APPS={"social_media":("social_media_dl_bytes","social_media_ul_bytes"),"youtube":("youtube_dl_bytes","youtube_ul_bytes"),"netflix":("netflix_dl_bytes","netflix_ul_bytes"),"google":("google_dl_bytes","google_ul_bytes"),"email":("email_dl_bytes","email_ul_bytes"),"gaming":("gaming_dl_bytes","gaming_ul_bytes"),"other":("other_dl_bytes","other_ul_bytes")}
def clean(raw):
    d=raw.copy(); d.columns=[str(c).strip().lower().replace(" ","_").replace("/","_") for c in d.columns]
    for c in d.select_dtypes(include="number"):
        if c not in ("msisdn_number","customer_id","imsi","imei","bearer_id"):
            d[c]=d[c].replace([np.inf,-np.inf],np.nan).fillna(d[c].median())
            q1,q3=d[c].quantile([.25,.75]); i=q3-q1
            if i>0:d[c]=d[c].clip(q1-1.5*i,q3+1.5*i)
    for c in d.select_dtypes(include=["object","string"]):
        m=d[c].mode(); d[c]=d[c].fillna(m.iloc[0] if len(m) else "Unknown")
    return d
def analyze(raw):
    d=clean(raw)
    ident="msisdn_number" if "msisdn_number" in d else "customer_id" if "customer_id" in d else None
    if ident is None: raise ValueError("Expected msisdn_number or customer_id column")
    d=d[d[ident].notna()].copy()
    if d.empty: raise ValueError("No customer IDs in source")
    duration="duration_s" if "duration_s" in d else "duration_ms" if "duration_ms" in d else None
    d["_duration"]=pd.to_numeric(d[duration],errors="coerce").fillna(0)/(1000 if duration=="duration_ms" else 1) if duration else 0
    for app,(dl,ul) in APPS.items():d[app+"_bytes"]=d.get(dl,0)+d.get(ul,0)
    d["_traffic"]=d.get("total_dl_bytes",0)+d.get("total_ul_bytes",0)
    d["_traffic"]=d["_traffic"].where(d["_traffic"]>0,d[[a+"_bytes" for a in APPS]].sum(axis=1))
    d["_tcp"]=d.get("tcp_dl_retrans_vol_bytes",0)+d.get("tcp_ul_retrans_vol_bytes",0)
    d["_rtt"]=pd.concat([d.get("avg_rtt_dl_ms",pd.Series(0,index=d.index)),d.get("avg_rtt_ul_ms",pd.Series(0,index=d.index))],axis=1).mean(axis=1)
    d["_tp"]=pd.concat([d.get("avg_bearer_tp_dl_kbps",pd.Series(0,index=d.index)),d.get("avg_bearer_tp_ul_kbps",pd.Series(0,index=d.index))],axis=1).mean(axis=1)
    agg={ "session_frequency":(ident,"size"),"session_duration_s":("_duration","sum"),"total_traffic_bytes":("_traffic","sum"),"tcp_retransmission_bytes":("_tcp","mean"),"avg_rtt_ms":("_rtt","mean"),"avg_throughput_kbps":("_tp","mean")}
    for a in APPS:agg[a+"_bytes"]=(a+"_bytes","sum")
    for c in ["handset_type","handset_manufacturer"]:
        if c in d:agg[c]=(c,lambda s:s.mode().iloc[0] if not s.mode().empty else "Unknown")
    customers=d.groupby(ident).agg(**agg).reset_index().rename(columns={ident:"customer_id"})
    if len(customers)<3: raise ValueError("At least three distinct customers are required for K-means")
    ecols=["session_frequency","session_duration_s","total_traffic_bytes"]; xcols=["tcp_retransmission_bytes","avg_rtt_ms","avg_throughput_kbps"]
    es=StandardScaler().fit_transform(customers[ecols]); em=KMeans(n_clusters=3,n_init=10,random_state=42).fit(es); least=np.argmin(em.cluster_centers_.mean(axis=1))
    xs=customers[xcols].copy(); xs["avg_throughput_kbps"]*=-1; xs=StandardScaler().fit_transform(xs)
    xm=KMeans(n_clusters=3,n_init=10,random_state=42).fit(xs); worst=np.argmax(xm.cluster_centers_.mean(axis=1))
    customers["engagement_cluster"]=em.labels_; customers["engagement_score"]=np.linalg.norm(es-em.cluster_centers_[least],axis=1)
    customers["experience_cluster"]=xm.labels_; customers["experience_score"]=np.linalg.norm(xs-xm.cluster_centers_[worst],axis=1)
    customers["satisfaction_score"]=(customers.engagement_score+customers.experience_score)/2
    sm=KMeans(n_clusters=2,n_init=10,random_state=42).fit(customers[["engagement_score","experience_score"]])
    customers["satisfaction_cluster"]=sm.labels_
    reg=Ridge().fit(customers[ecols+xcols],customers.satisfaction_score); pred=reg.predict(customers[ecols+xcols])
    customers["predicted_satisfaction_score"]=pred
    apps=pd.DataFrame({"application":list(APPS),"bytes":[customers[a+"_bytes"].sum() for a in APPS]})
    elbow={str(k):float(KMeans(n_clusters=k,n_init=10,random_state=42).fit(es).inertia_) for k in range(1,min(10,len(customers))+1)}
    return {"customers":customers,"scored":customers,"apps":apps,"elbow":elbow,"regression":{"rmse_train":float(mean_squared_error(customers.satisfaction_score,pred)**.5),"r2_train":float(r2_score(customers.satisfaction_score,pred)),"note":"In-sample diagnostic only; target is derived from same features."}}
