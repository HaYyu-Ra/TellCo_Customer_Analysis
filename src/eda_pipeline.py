# src/eda_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data_processing import clean_data

# Example pipeline for EDA
def eda_pipeline(df):
    pipeline = Pipeline([
        ('clean_data', clean_data(df)),
        ('scaler', StandardScaler())
    ])
    return pipeline.fit_transform(df)

def top_n_handsets(df, n=10):
    return df['Handset Type'].value_counts().head(n)
# src/user_overview_analysis.py

from src.eda_pipeline import top_n_handsets

def user_overview_analysis(df):
    # Top 10 handsets
    print("Top 10 Handsets:")
    print(top_n_handsets(df))

    # Top 3 manufacturers
    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    print("Top 3 Handset Manufacturers:", top_3_manufacturers)

    # Top 5 handsets for each of the top 3 manufacturers
    for manufacturer in top_3_manufacturers.index:
        top_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        print(f"Top 5 handsets for {manufacturer}:")
        print(top_handsets)
