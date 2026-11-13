from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from acis.analysis import load_data, aggregate_policies, hypothesis_tests, train_models

st.set_page_config(page_title="ACIS Insurance Analytics",layout="wide")
st.title("AlphaCare Insurance Solutions — Motor Portfolio Analytics")
st.caption("Historical exploratory analysis; not an approved pricing or underwriting tool.")
up=st.file_uploader("Upload MachineLearningRating_v3.txt (pipe-delimited)",type=["txt"])
default=Path("data/raw/MachineLearningRating_v3.txt")
if up:
    with tempfile.NamedTemporaryFile(suffix=".txt",delete=False) as f:
        f.write(up.getbuffer()); data=Path(f.name)
elif default.exists(): data=default
else:
    st.info("Upload the challenge dataset to begin. Raw data is intentionally excluded from Git.")
    st.stop()
rows=load_data(data); p=aggregate_policies(rows)
a,b,c=st.columns(3)
a.metric("Source rows",f"{len(rows):,}"); b.metric("Policies",f"{len(p):,}"); c.metric("Claim frequency",f"{p.has_claim.mean():.1%}")
t1,t2,t3=st.tabs(["Portfolio","Geography & tests","Model diagnostics"])
with t1:
    x,y=st.columns(2)
    for col,label,slot in [("premium","Policy premium",x),("claims","Policy claims",y)]:
        fig,ax=plt.subplots(); p[col].clip(lower=p[col].quantile(.01),upper=p[col].quantile(.99)).hist(ax=ax,bins=40); ax.set_title(label); slot.pyplot(fig); plt.close(fig)
with t2:
    tests,prov,postal=hypothesis_tests(p)
    fig,ax=plt.subplots(); prov.sort_values("claim_rate").plot.barh(x="province",y="claim_rate",ax=ax,legend=False); ax.set_xlabel("Policy claim frequency"); st.pyplot(fig); plt.close(fig)
    st.dataframe(prov,use_container_width=True); st.dataframe(postal.head(30),use_container_width=True); st.json(tests)
    st.caption("Postal-code comparisons include groups with at least 30 policies. Margin proxy is premium minus claims, not underwriting profit.")
with t3:
    st.write("Models use policy-level features and a random 80/20 holdout split.")
    if st.button("Train baseline models"): st.json(train_models(p))
    st.warning("Baseline scores are weak. No optimal-premium label or pricing objective is available; do not use this app to quote or change premiums.")
