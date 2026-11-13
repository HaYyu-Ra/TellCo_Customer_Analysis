from pathlib import Path
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from src.tellco.analytics import analyze

st.set_page_config(page_title="TellCo Network Intelligence", page_icon="📡", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem}.hero{padding:1.4rem 1.8rem;border-radius:16px;background:linear-gradient(120deg,#10243a,#17617a);color:white}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📡 TellCo Network Intelligence</h1><p>Customer overview · Engagement · Network experience · Satisfaction</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.header("Data source")
    up=st.file_uploader("Upload xDR data",type=["csv","xlsx","parquet"])
    try:
        secret_url=st.secrets.get("TELLCO_DATABASE_URL","")
    except Exception:
        secret_url=""
    db_url=st.text_input("PostgreSQL URL (optional)",value=secret_url or os.getenv("TELLCO_DATABASE_URL",""),type="password")
if up:
    name=up.name.lower()
    raw=pd.read_parquet(up) if name.endswith(".parquet") else pd.read_excel(up) if name.endswith(".xlsx") else pd.read_csv(up, dtype={"msisdn_number":"string","imsi":"string","imei":"string","bearer_id":"string"}, low_memory=False)
elif db_url:
    from sqlalchemy import create_engine
    raw=pd.read_sql_query("SELECT * FROM public.xdr_data",create_engine(db_url))
else:
    st.info("Upload an xDR CSV, Excel or Parquet extract, or configure TELLCO_DATABASE_URL in Streamlit Secrets. No sample results are fabricated.")
    st.code("pg_restore --no-owner --no-privileges -d tellco_analysis telecom.sql\n# Add TELLCO_DATABASE_URL to Streamlit Secrets",language="bash")
    st.stop()
try:
    result=analyze(raw); customers=result["customers"]; scored=result["scored"]; apps=result["apps"]
    a,b,c,d=st.columns(4)
    a.metric("Customers",f"{len(customers):,}")
    b.metric("xDR sessions",f"{len(raw):,}")
    c.metric("Total traffic",f"{customers.total_traffic_bytes.sum()/1e9:,.2f} GB")
    d.metric("Mean duration",f"{customers.session_duration_s.mean():,.1f} s")
    tabs=st.tabs(["Overview","Engagement","Experience","Satisfaction","Data quality"])
    with tabs[0]:
        l,r=st.columns(2)
        if "handset_type" in raw:
            l.plotly_chart(px.bar(raw.handset_type.value_counts().head(10).rename_axis("handset").reset_index(name="sessions"),x="handset",y="sessions",title="Top 10 handsets"),use_container_width=True)
        r.plotly_chart(px.bar(apps,x="application",y="bytes",title="Application traffic"),use_container_width=True)
        st.dataframe(apps,use_container_width=True,hide_index=True)
        st.caption("Application traffic sums download and upload bytes.")
    with tabs[1]:
        st.subheader("Customer engagement clusters")
        st.dataframe(scored.nlargest(10,"engagement_score"),use_container_width=True,hide_index=True)
        st.plotly_chart(px.scatter(scored,x="session_duration_s",y="total_traffic_bytes",color="engagement_cluster",hover_data=["customer_id"]),use_container_width=True)
        st.write("Elbow inertia",result["elbow"])
    with tabs[2]:
        st.subheader("Network experience")
        st.plotly_chart(px.scatter(customers,x="avg_rtt_ms",y="avg_throughput_kbps",color="tcp_retransmission_bytes",hover_data=["customer_id"]),use_container_width=True)
        if "handset_type" in customers:
            st.dataframe(customers.groupby("handset_type")[["avg_throughput_kbps","avg_rtt_ms","tcp_retransmission_bytes"]].mean().sort_values("avg_throughput_kbps",ascending=False),use_container_width=True)
    with tabs[3]:
        st.caption("Challenge-defined Euclidean distance proxies; not survey-based satisfaction.")
        st.dataframe(scored.nlargest(10,"satisfaction_score"),use_container_width=True,hide_index=True)
        st.plotly_chart(px.scatter(scored,x="engagement_score",y="experience_score",color="satisfaction_cluster"),use_container_width=True)
        st.write("Regression diagnostic",result["regression"])
        st.download_button("Download customer scores",scored.to_csv(index=False).encode(),"tellco_customer_scores.csv","text/csv")
    with tabs[4]:
        st.write({"rows":len(raw),"columns":len(raw.columns),"customers":len(customers)})
        st.dataframe(pd.DataFrame({"dtype":raw.dtypes.astype(str),"missing":raw.isna().sum(),"unique":raw.nunique()}),use_container_width=True)
except Exception as e:
    st.error(f"Unable to analyze this dataset: {e}")
