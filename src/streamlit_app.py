# src/streamlit_app.py
import sys
import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from src.config import db_url
from src.models import kmeans_clustering, aggregate_by_cluster
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_data_from_db(table_name):
    engine = create_engine(db_url)
    return pd.read_sql(table_name, con=engine)

def plot_top_handsets(df):
    if 'handset' in df.columns:
        top_10 = df['handset'].value_counts().head(10)
        st.bar_chart(top_10)
    else:
        st.write("Column 'handset' not found in the dataframe.")

def plot_correlation_heatmap(df):
    if not df.empty:
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()
    else:
        st.write("Dataframe is empty, cannot plot correlation heatmap.")

def main():
    st.title('TellCo Customer Analysis Dashboard')
    
    df = load_data_from_db('user_behavior')
    
    st.header('Cluster Analysis')
    clustered_df = kmeans_clustering(df)
    st.write(aggregate_by_cluster(clustered_df))
    
    st.header('Top 10 Handsets')
    plot_top_handsets(df)
    
    st.header('Correlation Heatmap')
    plot_correlation_heatmap(df)

if __name__ == '__main__':
    main()
