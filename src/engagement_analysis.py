# src/engagement_analysis.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(file_path):
    """Load the dataset from CSV."""
    return pd.read_csv(file_path)

def aggregate_metrics(df):
    """Aggregate metrics per customer."""
    aggregated_df = df.groupby('MSISDN').agg({
        'session_frequency': 'count',
        'session_duration': 'sum',
        'total_traffic': 'sum'
    }).reset_index()
    return aggregated_df

def normalize_metrics(df):
    """Normalize engagement metrics."""
    scaler = StandardScaler()
    metrics = ['session_frequency', 'session_duration', 'total_traffic']
    df[metrics] = scaler.fit_transform(df[metrics])
    return df

def perform_kmeans(df, k):
    """Perform K-means clustering."""
    kmeans = KMeans(n_clusters=k, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['session_frequency', 'session_duration', 'total_traffic']])
    return kmeans, df

def cluster_statistics(df):
    """Compute statistics for each cluster."""
    stats = df.groupby('cluster').agg({
        'session_frequency': ['min', 'max', 'mean', 'sum'],
        'session_duration': ['min', 'max', 'mean', 'sum'],
        'total_traffic': ['min', 'max', 'mean', 'sum']
    })
    return stats

def top_engaged_users(df):
    """Find top 10 engaged users per application."""
    return df.groupby('application').apply(lambda x: x.nlargest(10, 'total_traffic')).reset_index(drop=True)

def elbow_method(df):
    """Determine the optimal number of clusters using the elbow method."""
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[['session_frequency', 'session_duration', 'total_traffic']])
        inertia.append(kmeans.inertia_)
    return inertia
