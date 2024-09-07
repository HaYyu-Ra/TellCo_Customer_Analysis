from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def kmeans_clustering(df, n_clusters=3):
    """Apply KMeans clustering to the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with features for clustering.
        n_clusters (int): Number of clusters for KMeans.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Cluster' column indicating cluster assignments.
    """
    # Ensure required columns are present
    if not all(col in df.columns for col in ['xDR Session', 'Dur. (ms)', 'Total_Data (Bytes)']):
        raise ValueError("DataFrame must contain 'xDR Session', 'Dur. (ms)', and 'Total_Data (Bytes)' columns.")
    
    # Select features for clustering
    features = df[['xDR Session', 'Dur. (ms)', 'Total_Data (Bytes)']]
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    return df

def calculate_cluster_summary(df):
    """Calculate summary statistics for each cluster.

    Args:
        df (pd.DataFrame): DataFrame with cluster assignments.

    Returns:
        pd.DataFrame: Summary statistics for each cluster.
    """
    # Ensure 'Cluster' column is present
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column.")
    
    # Compute statistics for each cluster
    cluster_summary = df.groupby('Cluster').agg({
        'xDR Session': ['min', 'max', 'mean', 'sum'],
        'Dur. (ms)': ['min', 'max', 'mean', 'sum'],
        'Total_Data (Bytes)': ['min', 'max', 'mean', 'sum']
    })
    
    return cluster_summary
