import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Readiness Analyzer", layout="wide")
st.title("AI Readiness Analyzer – Flow 1")

st.markdown("""
### 📥 Input your organizational data
Please respond to the following based on your organization's current AI-related practices.
""")

# Mapping for inputs
training_map = {'No': 0, 'currently in development': 1, 'Yes': 2}
policies_map = {'No': 0, 'currently in development': 1, 'Yes': 2}
ethics_map = {'Not at all': 1, 'Minimally': 2, 'Moderately': 3, 'Extensively': 4}
productivity_map = {'0-20 %': 1, '20-40%': 2, '40-60%': 3, '60-80%': 4, '80-100%': 5}

# UI inputs
training_input = st.selectbox("**Has your organization provided any AI-related training or education programs for its employees?**", list(training_map.keys()))
policies_input = st.selectbox("**Has your organization implemented any policies or guidelines to address the ethical implications of AI use?**", list(policies_map.keys()))
ethics_input = st.selectbox("**To what extent has your organization considered the ethical implications of AI adoption (e.g. bias, transparency)?**", list(ethics_map.keys()))
integration_input = st.slider("**Rate the level of AI integration in daily operations (1=minimal, 10=extensive)**", 1, 10, 5)
observability_input = st.selectbox("**What percentage increase/decrease in productivity has been observed since the implementation of AI?**", list(productivity_map.keys()))
complexity_input = st.slider("**How would you rate the complexity of AI tools and systems used in your organization? (1=simple, 10=complex)**", 1, 10, 5)

# Convert to numeric
training_val = training_map.get(training_input.strip(), np.nan)
policies_val = policies_map.get(policies_input.strip(), np.nan)
ethics_val = ethics_map.get(ethics_input.strip(), np.nan)
observability_val = productivity_map.get(observability_input.strip(), np.nan)

# Calculate readiness score
values = [training_val, policies_val, ethics_val, integration_input, observability_val, complexity_input]
valid_values = [v for v in values if v is not None and not np.isnan(v)]
readiness_score = round(np.mean(valid_values), 2) if valid_values else 0.0

# Display result
st.markdown("""
### 📊 Estimated AI Readiness Score:
""")
st.success(f"{readiness_score}")

# Histogram (optional)
if 'show_hist' not in st.session_state:
    st.session_state.show_hist = False

if st.button("📈 Show Histogram of Readiness Score (Demo Sample)"):
    st.session_state.show_hist = not st.session_state.show_hist

if st.session_state.show_hist:
    # Fake data for demonstration
    demo_scores = np.random.normal(loc=readiness_score, scale=1, size=250)
    demo_scores = np.clip(demo_scores, 0, 5)
    plt.figure(figsize=(8, 5))
    sns.histplot(demo_scores, bins=10, kde=True, color='skyblue')
    plt.xlabel("AI Readiness Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of AI Readiness Scores (Synthetic Data)")
    st.pyplot(plt.gcf())
