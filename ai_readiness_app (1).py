import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import kruskal
from mord import LogisticAT  # ordinal logistic regression
import base64

st.set_page_config(layout="wide")
st.title("AI Readiness Framework Explorer")

# File upload
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")

    # Define readiness vars
    readiness_vars = [
        'JDR_AI_Training_Offered',
        'JDR_AI_Training_Hours',
        'TAM_AI_Integration_Level',
        'DOI_AI_Innovation_Competitiveness',
        'DOI_AI_Ethical_Policies',
        'JDR_AI_New_Jobs'
    ]

    df['AI_Readiness'] = df[readiness_vars].mean(axis=1)

    # Flow selection
    flow = st.sidebar.selectbox("Select flow", [
        "1. AI Readiness Evaluation",
        "2. Industry Benchmarking",
        "3. Employer Clustering",
        "4. Predictive Scenario (CLMM)",
        "5. Comparative Analysis"
    ])

    if flow.startswith("1"):
        st.header("AI Readiness Evaluation")
        # Clustering
        scaler = StandardScaler()
        X_clust = scaler.fit_transform(df[readiness_vars])
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X_clust)
        df['Cluster'] = kmeans.labels_

        pca = PCA(n_components=2)
        df[['PC1', 'PC2']] = pca.fit_transform(X_clust)

        st.subheader("PCA Projection with Cluster Assignments")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='husl', ax=ax)
        st.pyplot(fig)

        selected_idx = st.number_input("Row index for recommendation", 0, len(df)-1, 0)
        cluster = df.loc[selected_idx, 'Cluster']
        st.info(f"Your organization is in Cluster {cluster}. Suggested action: ...")

    elif flow.startswith("2"):
        st.header("Industry Benchmarking")
        if 'Industry_Label' not in df:
            st.warning("Missing 'Industry_Label' column.")
        else:
            scores = df.groupby("Industry_Label")['AI_Readiness'].mean().sort_values()
            fig, ax = plt.subplots()
            colors = sns.color_palette("Blues_r", len(scores))
            bars = ax.barh(scores.index, scores.values, color=colors)
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')
            ax.set_title("AI Readiness Score by Industry")
            st.pyplot(fig)

    elif flow.startswith("3"):
        st.header("Employer Clustering")
        heat_data = df.groupby("Cluster")[readiness_vars + ['AI_Readiness']].mean().T
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heat_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

        if 'Position' in df:
            df['Position_Label_Cleaned'] = df['Position'].replace({
                "Worker": "Non-management",
                "employee": "Non-management",
                "Lower or Operative Management": "Lower Management",
                "Middle Management": "Middle Management",
                "Top Management": "Top Management"
            })
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x='Position_Label_Cleaned', hue='Cluster', ax=ax2)
            st.pyplot(fig2)

    elif flow.startswith("4"):
        st.header("Predictive Scenario: Ordinal Logistic Model")
        if 'DOI_Future_Preparedness' not in df:
            st.warning("Missing outcome variable: 'DOI_Future_Preparedness'")
        else:
            try:
                model = LogisticAT(alpha=1.0)
                X = df[['AI_Readiness']].values
                y = df['DOI_Future_Preparedness'].astype(int)
                model.fit(X, y)
                pred = model.predict_proba(X[:1])
                st.write("Predicted probabilities for first case:")
                st.dataframe(pd.DataFrame(pred, columns=[f"Class {i+1}" for i in range(pred.shape[1])]))
            except Exception as e:
                st.error(f"Model error: {e}")

    elif flow.startswith("5"):
        st.header("Cluster Comparative Analysis")
        stat, pval = kruskal(*[group["AI_Readiness"] for _, group in df.groupby("Cluster")])
        st.write(f"Kruskal-Wallis Test: H = {stat:.3f}, p = {pval:.3f}")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Cluster', y='AI_Readiness', palette='Set2', ax=ax)
        means = df.groupby('Cluster')['AI_Readiness'].mean()
        for i, mean in enumerate(means):
            ax.plot(i, mean, 'ro')
        st.pyplot(fig)

    # Optional: Export current view to HTML
    if st.button("Export current report to HTML"):
        html_report = df.head(20).to_html()
        b64 = base64.b64encode(html_report.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="report.html">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)
