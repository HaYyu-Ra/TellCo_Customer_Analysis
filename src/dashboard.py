import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, euclidean_distances
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import pymysql
from sqlalchemy import create_engine


# Database URLs
postgres_db_url = "postgresql://postgres:admin@localhost:5432/tellco_analysis"
mysql_db_url = 'mysql+pymysql://root:@localhost:3306/user_engagement_satisfaction_db'
mysql_model_tracking_db_url = 'mysql+pymysql://root:@localhost:3306/model_tracking_db'


# Global data dictionary
data = {
    'Bearer Id': [1, 2, 3, 4, 5],
    'Start': ['2023-09-01 10:00:00', '2023-09-01 10:05:00', '2023-09-01 10:10:00', '2023-09-01 10:15:00', '2023-09-01 10:20:00'],
    'Start ms': [0, 0, 0, 0, 0],
    'End': ['2023-09-01 10:01:00', '2023-09-01 10:06:00', '2023-09-01 10:11:00', '2023-09-01 10:16:00', '2023-09-01 10:21:00'],
    'End ms': [60000, 60000, 60000, 60000, 60000],
    'Dur. (ms)': [60000, 60000, 60000, 60000, 60000],
    'IMSI': [123456789012345, 123456789012346, 123456789012347, 123456789012348, 123456789012349],
    'MSISDN/Number': [9876543210, 9876543211, 9876543212, 9876543213, 9876543214],
    'IMEI': [111111111111111, 111111111111112, 111111111111113, 111111111111114, 111111111111115],
    'Last Location Name': ['Location1', 'Location2', 'Location3', 'Location4', 'Location5'],
    'Avg RTT DL (ms)': [50, 60, 70, 80, 90],
    'Avg RTT UL (ms)': [40, 50, 60, 70, 80],
    'Avg Bearer TP DL (kbps)': [1000, 2000, 3000, 4000, 5000],
    'Avg Bearer TP UL (kbps)': [500, 1000, 1500, 2000, 2500],
    'TCP DL Retrans. Vol (Bytes)': [100, 200, 300, 400, 500],
    'TCP UL Retrans. Vol (Bytes)': [50, 100, 150, 200, 250],
    'DL TP < 50 Kbps (%)': [10, 20, 30, 40, 50],
    '50 Kbps < DL TP < 250 Kbps (%)': [5, 10, 15, 20, 25],
    '250 Kbps < DL TP < 1 Mbps (%)': [2, 4, 6, 8, 10],
    'DL TP > 1 Mbps (%)': [1, 2, 3, 4, 5],
    'UL TP < 10 Kbps (%)': [0.5, 1, 1.5, 2, 2.5],
    '10 Kbps < UL TP < 50 Kbps (%)': [0.2, 0.4, 0.6, 0.8, 1],
    '50 Kbps < UL TP < 300 Kbps (%)': [0.1, 0.2, 0.3, 0.4, 0.5],
    'UL TP > 300 Kbps (%)': [0.05, 0.1, 0.15, 0.2, 0.25],
    'HTTP DL (Bytes)': [1000, 2000, 3000, 4000, 5000],
    'HTTP UL (Bytes)': [500, 1000, 1500, 2000, 2500],
    'Activity Duration DL (ms)': [10000, 20000, 30000, 40000, 50000],
    'Activity Duration UL (ms)': [5000, 10000, 15000, 20000, 25000],
    'Dur. (ms).1': [60000, 60000, 60000, 60000, 60000],
    'Handset Manufacturer': ['Manufacturer1', 'Manufacturer2', 'Manufacturer3', 'Manufacturer4', 'Manufacturer5'],
    'Handset Type': ['Type1', 'Type2', 'Type3', 'Type4', 'Type5'],
    'Nb of sec with 125000B < Vol DL': [1, 2, 3, 4, 5],
    'Nb of sec with 1250B < Vol UL < 6250B': [0.5, 1, 1.5, 2, 2.5],
    'Nb of sec with 31250B < Vol DL < 125000B': [0.2, 0.4, 0.6, 0.8, 1],
    'Nb of sec with 37500B < Vol UL': [0.1, 0.2, 0.3, 0.4, 0.5],
    'Nb of sec with 6250B < Vol DL < 31250B': [0.05, 0.1, 0.15, 0.2, 0.25],
    'Nb of sec with 6250B < Vol UL < 37500B': [0.02, 0.04, 0.06, 0.08, 0.1],
    'Nb of sec with Vol DL < 6250B': [0.01, 0.02, 0.03, 0.04, 0.05],
    'Nb of sec with Vol UL < 1250B': [0.005, 0.01, 0.015, 0.02, 0.025],
    'Social Media DL (Bytes)': [100, 200, 300, 400, 500],
    'Social Media UL (Bytes)': [50, 100, 150, 200, 250],
    'Google DL (Bytes)': [1000, 2000, 3000, 4000, 5000],
    'Google UL (Bytes)': [500, 1000, 1500, 2000, 2500],
    'Email DL (Bytes)': [100, 200, 300, 400, 500],
    'Email UL (Bytes)': [50, 100, 150, 200, 250],
    'Youtube DL (Bytes)': [10000, 20000, 30000, 40000, 50000],
    'Youtube UL (Bytes)': [5000, 10000, 15000, 20000, 25000],
    'Netflix DL (Bytes)': [100000, 200000, 300000, 400000, 500000],
    'Netflix UL (Bytes)': [50000, 100000, 150000, 200000, 250000],
    'Gaming DL (Bytes)': [1000, 2000, 3000, 4000, 5000],
    'Gaming UL (Bytes)': [500, 1000, 1500, 2000, 2500],
    'Other DL (Bytes)': [100, 200, 300, 400, 500],
    'Other UL (Bytes)': [50, 100, 150, 200, 250],
    'Total UL (Bytes)': [1000000, 2000000, 3000000, 4000000, 5000000],
    'Total DL (Bytes)': [5000000, 10000000, 15000000, 20000000, 25000000]
}

# Convert global data dictionary to DataFrame
df_global_data = pd.DataFrame(data)

# Create SQLAlchemy engines
postgres_engine = create_engine(postgres_db_url)
mysql_engine = create_engine(mysql_db_url)
mysql_model_tracking_engine = create_engine(mysql_model_tracking_db_url)

def load_data():
    """Load data from the PostgreSQL database."""
    query = """
    SELECT "Bearer Id", "Start", "End", "Dur. (ms)", "IMSI", "MSISDN/Number", 
           "IMEI", "Last Location Name", "Avg RTT DL (ms)", "Avg Bearer TP DL (kbps)", 
           "TCP DL Retrans. Vol (Bytes)", "Handset Manufacturer", "Handset Type",
           "Social Media DL (Bytes)", "Youtube DL (Bytes)", "Netflix DL (Bytes)",
           "Google DL (Bytes)", "Email DL (Bytes)", "Gaming DL (Bytes)", "Other DL (Bytes)"
    FROM xdr_data
    """
    return pd.read_sql(query, postgres_engine)

def preprocess_data(df):
    """Preprocess data: Handle missing values and outliers."""
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)
    df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
    return df

def plot_top_applications(df):
    """Plot Top 3 Most Used Applications"""
    st.subheader("Top 3 Most Used Applications")
    app_totals = df[['Social Media DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)']].sum()
    fig, ax = plt.subplots()
    app_totals.plot(kind='bar', ax=ax)
    ax.set_xlabel('Application')
    ax.set_ylabel('Total Traffic')
    st.pyplot(fig)

def task1_analysis(df):
    st.header("Task 1: User Overview Analysis")

    # Task 1.1: Aggregate Information
    st.subheader("Task 1.1: Aggregate Information")
    agg_df = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',
        'TCP DL Retrans. Vol (Bytes)': 'sum',
        'Avg RTT DL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Handset Type': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }).reset_index()
    st.write(agg_df)

    # Visualization: Distribution of Duration
    st.subheader("Distribution of Duration (ms)")
    plt.figure(figsize=(10, 6))
    sns.histplot(agg_df['Dur. (ms)'], bins=30, kde=True)
    plt.title("Distribution of Duration (ms)")
    plt.xlabel("Duration (ms)")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Task 1.2: Top 10 Handsets
    st.subheader("Task 1.2: Top 10 Handsets")
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    st.write(top_10_handsets)

    # Visualization: Top 10 Handsets
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, palette="viridis")
    plt.title("Top 10 Handsets")
    plt.xlabel("Count")
    plt.ylabel("Handset Type")
    st.pyplot(plt)

    # Task 1.3: Top 3 Handset Manufacturers
    st.subheader("Task 1.3: Top 3 Handset Manufacturers")
    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    st.write(top_3_manufacturers)

    # Visualization: Top 3 Handset Manufacturers
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_3_manufacturers.values, y=top_3_manufacturers.index, palette="viridis")
    plt.title("Top 3 Handset Manufacturers")
    plt.xlabel("Count")
    plt.ylabel("Manufacturer")
    st.pyplot(plt)

    # Task 1.4: Top 5 Handsets per Top 3 Manufacturers
    st.subheader("Task 1.4: Top 5 Handsets per Top 3 Manufacturers")
    for manufacturer in top_3_manufacturers.index:
        top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        st.write(f"Top 5 Handsets for {manufacturer}:")
        st.write(top_5_handsets)

        # Visualization: Top 5 Handsets per Manufacturer
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_5_handsets.values, y=top_5_handsets.index, palette="viridis")
        plt.title(f"Top 5 Handsets for {manufacturer}")
        plt.xlabel("Count")
        plt.ylabel("Handset Type")
        st.pyplot(plt)
    
    st.subheader("Recommendations")
    st.write("""
        Based on the analysis, we recommend the marketing team focus on promoting the top handset models 
        from the leading manufacturers. Additionally, tailored marketing campaigns can be developed for users 
        with the most used devices.
    """)

def task2_analysis(df):
    st.header("Task 2: User Engagement Analysis")

    # Display the DataFrame columns
    st.subheader("DataFrame Columns")
    st.write(df.columns.tolist())

    # Task 2.1: Top 10 Customers per Engagement Metric
    st.subheader("Task 2.1: Top 10 Customers per Engagement Metric")
    engagement_metrics = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',
        'TCP DL Retrans. Vol (Bytes)': 'sum',
        'Avg Bearer TP DL (kbps)': 'mean'
    }).reset_index()

    # Top 10 by Duration
    st.write("Top 10 Customers by Duration")
    top_duration = engagement_metrics.nlargest(10, 'Dur. (ms)')
    st.write(top_duration)

    fig, ax = plt.subplots()
    sns.barplot(x='MSISDN/Number', y='Dur. (ms)', data=top_duration, ax=ax)
    ax.set_title('Top 10 Customers by Duration')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Top 10 by Download Volume
    st.write("Top 10 Customers by Download Volume")
    top_volume = engagement_metrics.nlargest(10, 'TCP DL Retrans. Vol (Bytes)')
    st.write(top_volume)

    fig, ax = plt.subplots()
    sns.barplot(x='MSISDN/Number', y='TCP DL Retrans. Vol (Bytes)', data=top_volume, ax=ax)
    ax.set_title('Top 10 Customers by Download Volume')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Top 10 by Throughput
    st.write("Top 10 Customers by Throughput")
    top_throughput = engagement_metrics.nlargest(10, 'Avg Bearer TP DL (kbps)')
    st.write(top_throughput)

    fig, ax = plt.subplots()
    sns.barplot(x='MSISDN/Number', y='Avg Bearer TP DL (kbps)', data=top_throughput, ax=ax)
    ax.set_title('Top 10 Customers by Throughput')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Task 2.2: K-Means Clustering for User Engagement
    st.subheader("Task 2.2: K-Means Clustering for User Engagement")
    metrics = ['Dur. (ms)', 'TCP DL Retrans. Vol (Bytes)', 'Avg Bearer TP DL (kbps)']
    normalized_metrics = (engagement_metrics[metrics] - engagement_metrics[metrics].mean()) / engagement_metrics[metrics].std()

    kmeans = KMeans(n_clusters=3, random_state=0).fit(normalized_metrics)
    engagement_metrics['Engagement Cluster'] = kmeans.labels_

    st.write("Cluster Centers:")
    st.write(pd.DataFrame(kmeans.cluster_centers_, columns=metrics))

    st.write("Engagement Cluster Distribution")
    st.write(engagement_metrics.groupby('Engagement Cluster').mean())

    # Visualizing clusters
    fig, ax = plt.subplots()
    sns.scatterplot(data=engagement_metrics, x='Dur. (ms)', y='TCP DL Retrans. Vol (Bytes)', hue='Engagement Cluster', palette='viridis', ax=ax)
    ax.set_title('Engagement Clusters')
    st.pyplot(fig)
def task3_analysis(df):
    st.header("Task 3: User Experience Analysis")

    # Task 3.1: Aggregate metrics for user experience
    st.subheader("Task 3.1: Aggregate Metrics for User Experience")

    # Handling missing values and outliers
    imputer = SimpleImputer(strategy='mean')
    df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']] = imputer.fit_transform(
        df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']]
    )

    user_experience_df = df.groupby('MSISDN/Number').agg({
        'Avg RTT DL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'TCP DL Retrans. Vol (Bytes)': 'sum',
        'Handset Type': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }).reset_index()

    # Display summary statistics
    st.write(user_experience_df.describe())

    # Task 3.2: Perform linear regression to analyze the relationship
    st.subheader("Task 3.2: Linear Regression to Analyze Experience")

    # Ensure 'Dur. (ms)' column exists and is properly aggregated
    if 'Dur. (ms)' not in df.columns:
        st.error("'Dur. (ms)' column is missing from the dataframe.")
        return

    # Preparing data for linear regression
    X = user_experience_df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']]
    y = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().values

    # Check if the lengths of X and y match
    if len(X) != len(y):
        st.error("Mismatch between feature set X and target variable y.")
        return

    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared:", r2_score(y_test, y_pred))

    # Save the model
    model_filename = "linear_regression_model.pkl"
    joblib.dump(model, model_filename)
    st.write(f"Model saved as {model_filename}")

    # Visualize the relationship
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Duration")
    ax.set_ylabel("Predicted Duration")
    ax.set_title("Actual vs. Predicted Duration")
    st.pyplot(fig)

    # Task 3.4 - K-means Clustering
    st.subheader("Task 3.4: K-means Clustering")

    # Prepare data for clustering
    X_clustering = user_experience_df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']]
    scaler = StandardScaler()
    X_clustering_scaled = scaler.fit_transform(X_clustering)

    kmeans = KMeans(n_clusters=3, random_state=0)
    user_experience_df['cluster'] = kmeans.fit_predict(X_clustering_scaled)

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Avg RTT DL (ms)', y='Avg Bearer TP DL (kbps)', hue='cluster', palette='viridis', data=user_experience_df, ax=ax)
    ax.set_title('User Clusters Based on Experience Metrics')
    ax.set_xlabel('Avg RTT DL (ms)')
    ax.set_ylabel('Avg Bearer TP DL (kbps)')
    st.pyplot(fig)


def task4_analysis(df):
    st.header("Task 4: User Satisfaction Analysis")

    # Example data
    user_data = pd.DataFrame(data)  # Ensure 'data' is defined or loaded

    # Calculate satisfaction scores
    engagement_cluster_centers = np.array([[5000000]])
    experience_cluster_centers = np.array([[25000000]])

    def calculate_euclidean_distance(point, cluster_center):
        return np.linalg.norm(point - cluster_center)

    user_data['engagement_score'] = user_data['Total UL (Bytes)'].apply(lambda x: calculate_euclidean_distance([x], engagement_cluster_centers[0]))
    user_data['experience_score'] = user_data['Total DL (Bytes)'].apply(lambda x: calculate_euclidean_distance([x], experience_cluster_centers[0]))
    user_data['satisfaction_score'] = (user_data['engagement_score'] + user_data['experience_score']) / 2

    # Task 4.3 - Clustering Users Based on Satisfaction Scores
    satisfaction_scores = user_data[['satisfaction_score']]
    scaler = StandardScaler()
    satisfaction_scores_scaled = scaler.fit_transform(satisfaction_scores)

    kmeans = KMeans(n_clusters=3, random_state=0)
    user_data['cluster'] = kmeans.fit_predict(satisfaction_scores_scaled)

    # Task 4.4 - Visualization of User Clusters
    st.subheader('User Clusters Based on Satisfaction Scores')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Total UL (Bytes)', y='Total DL (Bytes)', hue='cluster', palette='viridis', data=user_data, ax=ax)
    ax.set_title('User Clusters Based on Satisfaction Scores')
    ax.set_xlabel('Total UL (Bytes)')
    ax.set_ylabel('Total DL (Bytes)')
    st.pyplot(fig)

    # Task 4.5 - Linear Regression Analysis
    X = user_data[['Total UL (Bytes)']]
    y = user_data['satisfaction_score']

    regressor = LinearRegression()
    regressor.fit(X, y)
    user_data['predicted_satisfaction_score'] = regressor.predict(X)

    st.subheader('Linear Regression Analysis of Satisfaction Scores')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(user_data['Total UL (Bytes)'], y, color='blue', label='Actual Satisfaction Scores')
    ax.plot(user_data['Total UL (Bytes)'], user_data['predicted_satisfaction_score'], color='red', linewidth=2, label='Fitted Line')
    ax.set_title('Linear Regression Analysis of Satisfaction Scores')
    ax.set_xlabel('Total UL (Bytes)')
    ax.set_ylabel('Satisfaction Score')
    ax.legend()
    st.pyplot(fig)

    # Print the top 10 satisfied customers
    top_10_satisfied_customers = user_data.nlargest(10, 'satisfaction_score')
    st.subheader('Top 10 Satisfied Customers')
    st.write(top_10_satisfied_customers[['Bearer Id', 'satisfaction_score']])

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Task 1", "Task 2", "Task 3", "Task 4"])
    
    df = load_data()  # Ensure you load your data here

    if selection == "Task 1":
        task1_analysis(df)
    elif selection == "Task 2":
        task2_analysis(df)
    elif selection == "Task 3":
        task3_analysis(df)
    elif selection == "Task 4":
        task4_analysis(df)

if __name__ == "__main__":
    main()
