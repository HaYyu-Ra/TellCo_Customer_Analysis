import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def user_overview_analysis(df):
    st.header("User Overview Analysis")

    # Define required columns based on provided schema
    required_columns = [
        'Total DL (Bytes)', 'Total UL (Bytes)', 'Bearer Id', 'Dur. (ms)',
        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'HTTP DL (Bytes)',
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)'
    ]

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns in the DataFrame: {', '.join(missing_columns)}")
        return

    # Aggregated User Data
    try:
        user_agg_df = df.groupby('Bearer Id').agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum',
            'Dur. (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'HTTP DL (Bytes)': 'sum',
            'Social Media DL (Bytes)': 'sum',
            'Google DL (Bytes)': 'sum',
            'Email DL (Bytes)': 'sum',
            'Youtube DL (Bytes)': 'sum',
            'Netflix DL (Bytes)': 'sum',
            'Gaming DL (Bytes)': 'sum'
        }).reset_index()

        user_agg_df['Total_Data (Bytes)'] = user_agg_df['Total DL (Bytes)'] + user_agg_df['Total UL (Bytes)']

        # Handle missing values and outliers
        user_agg_df.fillna(user_agg_df.mean(), inplace=True)
        user_agg_df = user_agg_df[user_agg_df['Total DL (Bytes)'] <= user_agg_df['Total DL (Bytes)'].quantile(0.95)]
    except KeyError as e:
        st.error(f"KeyError: {e}")
        return

    # Top 10 Handsets Used by Customers
    st.subheader("Top 10 Handsets Used by Customers")
    if 'Handset Type' in df.columns:
        top_handsets = df['Handset Type'].value_counts().head(10)
        st.bar_chart(top_handsets)
    else:
        st.warning("Handset data not available.")

    # Top 3 Handset Manufacturers
    st.subheader("Top 3 Handset Manufacturers")
    if 'Handset Manufacturer' in df.columns:
        top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
        st.bar_chart(top_manufacturers)
    else:
        st.warning("Handset manufacturer data not available.")

    # Top 5 Handsets per Top 3 Manufacturers
    st.subheader("Top 5 Handsets per Top 3 Manufacturer")
    if 'Handset Manufacturer' in df.columns and 'Handset Type' in df.columns:
        top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3).index
        top_handsets_per_manufacturer = df[df['Handset Manufacturer'].isin(top_3_manufacturers)]
        top_handsets_per_manufacturer = top_handsets_per_manufacturer.groupby(['Handset Manufacturer', 'Handset Type']).size().unstack()
        st.write(top_handsets_per_manufacturer)
    else:
        st.warning("Handset and/or handset manufacturer data not available.")

    # Aggregated Metrics
    st.subheader("Aggregated Metrics")
    decile_summary = user_agg_df.groupby(pd.qcut(user_agg_df['Total DL (Bytes)'], 10)).agg({
        'Total_Data (Bytes)': 'sum'
    }).reset_index()
    st.write("Decile Summary:")
    st.write(decile_summary)

    # Univariate Analysis
    st.subheader("Univariate Analysis")

    # Distribution of Session Duration
    st.subheader("Distribution of Duration")
    fig, ax = plt.subplots()
    sns.histplot(user_agg_df['Dur. (ms)'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Duration (ms)')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Distribution of Total Data
    st.subheader("Distribution of Total Data")
    fig, ax = plt.subplots()
    sns.histplot(user_agg_df['Total_Data (Bytes)'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Total Data')
    ax.set_xlabel('Total Data (Bytes)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Bivariate Analysis
    st.subheader("Session Duration vs. Total Data")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Dur. (ms)', y='Total_Data (Bytes)', data=user_agg_df, ax=ax)
    ax.set_title('Session Duration vs. Total Data')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Total Data (Bytes)')
    st.pyplot(fig)
