import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

st.title("AI Readiness Evaluation App")

st.sidebar.header("Upload your dataset")
file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("File uploaded successfully!")
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    st.sidebar.header("Select target group")
    group = st.sidebar.selectbox("Target group:", ["Employees", "Employers"])

    if group == "Employees":
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
        st.error(f"The following expected columns are missing from your data: {missing_cols}")
    else:
        df['AI_Readiness'] = df[readiness_vars].mean(axis=1)

        st.subheader("📊 AI Readiness Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['AI_Readiness'], kde=True, color="skyblue", edgecolor="black", bins=10, ax=ax)
        ax.set_xlabel("AI Readiness Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.subheader("📈 PCA Visualization and Clustering")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[readiness_vars])

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters

        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df['PC1'] = components[:, 0]
        df['PC2'] = components[:, 1]

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, edgecolor="black", alpha=0.8)
        ax2.set_title("PCA Projection with Cluster Assignments")
        st.pyplot(fig2)

        st.subheader("🧠 Personalized Evaluation")
        st.markdown("Answer a few questions to estimate your organization's AI Readiness score:")

        with st.form("readiness_form"):
            q1 = st.selectbox("Has your organization provided AI-related training?", ["Yes", "No", "currently in development"])
            q2 = st.slider("To what extent has your organization considered ethical AI implications?", 1, 4, 2)
            q3 = st.selectbox("Has your organization implemented ethical AI policies?", ["Yes", "No", "currently in development"])
            q4 = st.slider("Rate the level of AI integration (1=Minimal, 10=Extensive)", 1, 10, 5)
            q5 = st.selectbox("Productivity gain from AI", ['0-20 %', '20-40%', '40-60%', '60-80%', '80-100%'])
            q6 = st.slider("Complexity of AI tools (1=Simple, 10=Complex)", 1, 10, 5)
            submit = st.form_submit_button("Calculate Readiness Score")

        if submit:
            map_train = {"No": 0, "currently in development": 1, "Yes": 2}
            map_prod = {'0-20 %': 1, '20-40%': 2, '40-60%': 3, '60-80%': 4, '80-100%': 5}

            response_dict = {
                'JDR_Job_Resources_Training': map_train[q1],
                'DOI_Ethics_Consideration': q2,
                'DOI_Ethics_Policies': map_train[q3],
                'TAM_Integration_Level': q4,
                'DOI_Observability': map_prod[q5],
                'TAM_Complexity': q6
            }

            df_temp = pd.DataFrame([response_dict])
            scaler_input = MinMaxScaler()
            readiness = scaler_input.fit_transform(df_temp)
            readiness_score = np.mean(readiness)

            st.success(f"Estimated AI Readiness Score: {readiness_score:.2f}")

            # Compare with population
            fig3, ax3 = plt.subplots()
            sns.histplot(df['AI_Readiness'], bins=10, kde=True, color='lightblue', label='Population', ax=ax3)
            ax3.axvline(readiness_score, color='red', linestyle='--', label='Your score')
            ax3.legend()
            ax3.set_title("Your Score vs Distribution")
            st.pyplot(fig3)
