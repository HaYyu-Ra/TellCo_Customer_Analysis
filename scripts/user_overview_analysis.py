import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text

# Configuration for database
db_url = 'postgresql://postgres:admin@localhost:5432/tellco_analysis'

# Function to load data from PostgreSQL
def load_data_from_db(query):
    """Load data from PostgreSQL database into DataFrame."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df

# Load dataset
def load_dataset():
    """Load the dataset from PostgreSQL."""
    query = "SELECT * FROM xdr_data"  # Update this query based on your table schema
    return load_data_from_db(query)

# Task 1 - Bearer Overview Analysis
def bearer_overview_analysis(df):
    """Perform Bearer Overview Analysis on the given DataFrame."""
    # Task 1.1 - Aggregate per Bearer Id
    bearer_agg_df = df.groupby('Bearer Id').agg({
        'Dur. (ms)': 'sum',
        'Total_DL (Bytes)': 'sum'
    }).reset_index()

    # Task 1.2 - Exploratory Data Analysis
    # Replace missing values
    bearer_agg_df.fillna(bearer_agg_df.mean(), inplace=True)

    # Segment users into deciles based on session duration
    bearer_agg_df['decile'] = pd.qcut(bearer_agg_df['Dur. (ms)'], 10, labels=False)
    decile_summary = bearer_agg_df.groupby('decile').agg({
        'Total_DL (Bytes)': 'sum'
    }).reset_index()

    # Basic Metrics
    basic_metrics = bearer_agg_df.describe()

    # Univariate Analysis
    # Non-Graphical
    dispersion_params = bearer_agg_df[['Dur. (ms)', 'Total_DL (Bytes)']].agg(['mean', 'median', 'std', 'var'])
    
    # Graphical
    plt.figure(figsize=(12, 6))
    sns.histplot(bearer_agg_df['Dur. (ms)'], bins=30, kde=True)
    plt.title('Distribution of Session Duration')
    plt.xlabel('Session Duration')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(bearer_agg_df['Total_DL (Bytes)'], bins=30, kde=True)
    plt.title('Distribution of Total Data')
    plt.xlabel('Total Data (Bytes)')
    plt.ylabel('Frequency')
    plt.show()

    # Bivariate Analysis
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Dur. (ms)', y='Total_DL (Bytes)', data=bearer_agg_df)
    plt.title('Session Duration vs. Total Data')
    plt.xlabel('Session Duration')
    plt.ylabel('Total Data (Bytes)')
    plt.show()

    # Correlation Analysis
    corr_matrix = bearer_agg_df[['Total_DL (Bytes)']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    # Dimensionality Reduction - PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(bearer_agg_df[['Dur. (ms)', 'Total_DL (Bytes)']])
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    plt.title('PCA of Session Duration and Total Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return {
        'basic_metrics': basic_metrics,
        'dispersion_params': dispersion_params,
        'decile_summary': decile_summary,
        'corr_matrix': corr_matrix,
        'pca_df': pca_df
    }

# Main function to run the analysis
def main():
    df = load_dataset()
    results = bearer_overview_analysis(df)
    
    # Output results
    print("Basic Metrics:")
    print(results['basic_metrics'])
    
    print("Dispersion Parameters:")
    print(results['dispersion_params'])
    
    print("Decile Summary:")
    print(results['decile_summary'])
    
    print("Correlation Matrix:")
    print(results['corr_matrix'])
    
    print("PCA Results:")
    print(results['pca_df'].head())

if __name__ == "__main__":
    main()
