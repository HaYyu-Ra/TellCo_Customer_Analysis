import pandas as pd

def create_user_metrics(df):
    """
    Creates features related to user engagement metrics like total sessions, total traffic, etc.
    
    Args:
        df (pd.DataFrame): Data containing xDR session information.
    
    Returns:
        pd.DataFrame: Data with new features.
    """
    # Total session count per user
    user_metrics = df.groupby('MSISDN').agg({
        'session_id': 'count',       # Total number of sessions
        'session_duration': 'sum',   # Total session duration
        'total_data': 'sum',         # Total data (DL + UL)
    }).reset_index()

    # Rename columns for better readability
    user_metrics.rename(columns={
        'session_id': 'total_sessions',
        'session_duration': 'total_duration',
        'total_data': 'total_data_usage'
    }, inplace=True)

    return user_metrics

def create_deciles(df, column):
    """
    Segments users into deciles based on a specific column.
    
    Args:
        df (pd.DataFrame): DataFrame with user metrics.
        column (str): Column on which deciles are to be created.
    
    Returns:
        pd.DataFrame: Data with a new 'decile' column.
    """
    df['decile'] = pd.qcut(df[column], 10, labels=False) + 1
    return df

def feature_scaling(df, feature_columns):
    """
    Normalizes the specified columns for further analysis (e.g., clustering).
    
    Args:
        df (pd.DataFrame): Data containing features to be scaled.
        feature_columns (list): List of columns to scale.
    
    Returns:
        pd.DataFrame: Data with scaled features.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df
