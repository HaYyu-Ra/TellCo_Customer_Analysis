# src/user_engagement_dashboard.py

import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import your module
from engagement_analysis import (
    load_data,
    aggregate_metrics,
    normalize_metrics,
    perform_kmeans,
    cluster_statistics,
    top_engaged_users,
    elbow_method
)

def main():
    st.title("User Engagement Dashboard")

    # Database URL and table name
    DATABASE_URL = "postgresql://postgres:admin@localhost:5432/tellco_analysis"
    TABLE_NAME = "xdr_data"
    
    # Load data from PostgreSQL
    df = load_data(DATABASE_URL, TABLE_NAME)
    
    if df.empty:
        st.error("No data found or failed to load data. Please check the database connection and table name.")
        return

    # Aggregate metrics
    agg_df = aggregate_metrics(df)

    # Normalize metrics
    norm_df = normalize_metrics(agg_df)

    # K-means clustering
    kmeans, clustered_df = perform_kmeans(norm_df, k=3)

    # Display clustering results
    st.subheader("Cluster Statistics")
    stats = cluster_statistics(clustered_df)
    st.write(stats)

    # Plot top engaged users per application
    st.subheader("Top Engaged Users Per Application")
    top_users_df = top_engaged_users(df)
    st.write(top_users_df)

    # Plot top 3 most used applications
    st.subheader("Top 3 Most Used Applications")
    top_apps = df['application'].value_counts().head(3)
    st.bar_chart(top_apps)

    # Elbow Method for optimal k
    st.subheader("Elbow Method for Optimal k")
    inertia = elbow_method(norm_df)
    plt.figure()
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    st.pyplot(plt)

if __name__ == "__main__":
    main()
