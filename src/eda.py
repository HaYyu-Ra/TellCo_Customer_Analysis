import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from src.config import db_url

def load_data_from_db(table_name):
    engine = create_engine(db_url)
    return pd.read_sql(table_name, con=engine)

# Basic statistical description
def basic_statistics(df):
    return df.describe()

# Univariate Analysis - Plot histograms
def plot_histograms(df, columns):
    df[columns].hist(bins=20, figsize=(12, 8))
    plt.show()

# Correlation matrix
def plot_correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

if __name__ == '__main__':
    df = load_data_from_db('user_behavior')
    
    # Basic Stats
    print(basic_statistics(df))
    
    # Plot Univariate Analysis
    plot_histograms(df, ['total_DL', 'total_UL', 'total_volume'])
    
    # Correlation Matrix
    plot_correlation_matrix(df)
