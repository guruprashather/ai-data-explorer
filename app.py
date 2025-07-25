import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from eda_report import generate_sweetviz_report
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="AI-Driven Data Explorer", layout="wide")
st.title("🚀 AI-Driven Data Explorer")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show Data
    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Basic Summary
    st.subheader("📊 Data Summary")
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Visualizations
    if st.button("Show Visualizations"):
        st.subheader("🔍 Histograms")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            st.write(f"📈 Histogram of {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    # Feature Selection
    st.subheader("🧠 Feature Selection (Top 3 using ANOVA)")
    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Select Top Features"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical
        X = pd.get_dummies(X)
        X = X.fillna(0)

        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=3)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        st.write("Top 3 Selected Features:", list(selected_features))

    # Sweetviz Report
    if st.button("Generate Full EDA Report"):
        report_path = generate_sweetviz_report(df)
        st.success("📑 Sweetviz report generated!")

        # Show report in Streamlit
        with open(report_path, 'r') as f:
            html_data = f.read()
            components.html(html_data, height=1000, scrolling=True)

        # Download button
        with open(report_path, 'rb') as f:
            btn = st.download_button(
                label="📥 Download EDA Report",
                data=f,
                file_name="sweetviz_report.html",
                mime="text/html"
            )

else:
    st.info("📂 Please upload a CSV file to begin.")
