from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def kmeans_clustering(df, n_clusters=3):
    """
    Perform KMeans clustering on the dataset.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe with columns for clustering.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'cluster' column for cluster labels.
    """
    # Check if required columns exist
    required_columns = ['sessions', 'session_duration', 'total_data']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataframe.")
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[required_columns])
    
    return df

def elbow_method(df):
    """
    Determine the optimal number of clusters using the elbow method.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe with columns for clustering.

    Returns:
    - dict: A dictionary with the number of clusters as keys and silhouette scores as values.
    """
    # Check if required columns exist
    required_columns = ['sessions', 'session_duration', 'total_data']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataframe.")
    
    # Compute optimal k using the elbow method
    scores = {}
    for k in range(2, 11):  # Checking from 2 to 10 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[required_columns])
        labels = kmeans.labels_
        if len(set(labels)) > 1:  # Ensure more than one cluster
            score = silhouette_score(df[required_columns], labels)
            scores[k] = score
        else:
            scores[k] = None  # If only one cluster, silhouette score is not meaningful

    return scores
