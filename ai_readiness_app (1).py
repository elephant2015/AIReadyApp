import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import kruskal

st.set_page_config(layout="wide", page_title="AI Readiness App")

st.title("🧠 AI Readiness Assessment and Insights")

uploaded_file = st.file_uploader("Upload your file", type=["csv"])

if uploaded_file is not None:
    user_type = st.radio("Select respondent type:", ["Employees", "Employers"])
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if user_type == "Employees":
        readiness_vars = [
            'JDR_Job_Resources_Training',
            'DOI_Ethics_Policies',
            'DOI_Ethics_Consideration',
            'TAM_Integration_Level',
            'DOI_Observability',
            'TAM_Complexity'
        ]
    else:
        readiness_vars = [
            'JDR_AI_Training_Offered',
            'JDR_AI_Training_Hours',
            'TAM_AI_Integration_Level',
            'DOI_AI_Innovation_Competitiveness',
            'DOI_AI_Ethical_Policies',
            'JDR_AI_New_Jobs'
        ]

    missing_cols = [col for col in readiness_vars if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for readiness computation: {missing_cols}")
        st.stop()

    df['AI_Readiness'] = df[readiness_vars].mean(axis=1)

    st.subheader("📊 AI Readiness Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['AI_Readiness'], bins=10, kde=True, color='skyblue')
    plt.xlabel("AI Readiness Score")
    plt.ylabel("Frequency")
    st.pyplot(fig)

    # Clustering
    st.subheader("🔍 Clustering Based on AI Readiness and Automation Risk (if available)")
    cluster_features = ['AI_Readiness']
    if 'Automation_Risk' in df.columns:
        cluster_features.append('Automation_Risk')

    X = df[cluster_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA Projection
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df['PC1'] = components[:, 0]
    df['PC2'] = components[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, edgecolor='black')
    plt.title("PCA Projection with Cluster Assignments")
    st.pyplot(fig)

    # Boxplot
    st.subheader("📦 AI Readiness by Cluster")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Cluster', y='AI_Readiness', palette='Set3')
    plt.title("AI Readiness Distribution by Cluster")
    st.pyplot(fig)

    # Kruskal-Wallis
    st.subheader("📌 Kruskal-Wallis Test for AI Readiness Differences Across Clusters")
    groups = [group["AI_Readiness"].values for _, group in df.groupby("Cluster")]
    stat, p = kruskal(*groups)
    st.write(f"Kruskal-Wallis H-statistic: {stat:.3f}, p-value: {p:.4f}")

    # Heatmap for variable means
    st.subheader("📈 Cluster Profile Heatmap")
    cluster_means = df.groupby("Cluster")[readiness_vars + ['AI_Readiness']].mean().T
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".2f")
    st.pyplot(fig)

    # Optional: Export HTML report
    if st.button("Export Cluster Summary Table to CSV"):
        cluster_means.to_csv("cluster_summary.csv")
        st.success("Exported cluster_summary.csv")

else:
    st.info("Please upload a file to begin.")
