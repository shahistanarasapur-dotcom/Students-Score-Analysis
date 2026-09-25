Here is the complete Python code for your Streamlit web application. It parses the analysis from your `NaviBayes.ipynb` notebook and converts it into a fully dynamic, interactive data science dashboard using Streamlit, Plotly, and Pandas.

### `app.py`
<img width="1600" height="900" alt="Screenshot 2026-09-25 111753" src="https://github.com/user-attachments/assets/49ddd806-f3d8-4833-996f-414e56651c0a" />
<img width="1600" height="900" alt="Screenshot 2026-09-25 111649" src="https://github.com/user-attachments/assets/c9fbb97b-945e-44a5-ac53-1af9d64a836c" />
```python
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

# 1. Page Configuration
st.set_page_config(
    page_title="NaviBayes - Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Data Loading & Caching
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    # Fallback default dataset structure matching NaviBayes.ipynb
    np.random.seed(42)
    n_samples = 399
    data = {
        "Math_Score": np.random.randint(60, 101, n_samples),
        "Reading_Score": np.random.randint(60, 101, n_samples),
        "Writing_Score": np.random.randint(60, 101, n_samples),
        "Placement_Score": np.random.randint(60, 101, n_samples),
        "Club_Join_Date": np.random.choice(
            [2018, 2019, 2020, 2021], n_samples, p=[0.25, 0.25, 0.25, 0.25]
        ),
    }
    return pd.DataFrame(data)


# 3. Sidebar Setup
st.sidebar.title("🎓 NaviBayes Control Panel")
uploaded_file = st.sidebar.file_uploader(
    "Upload 'Students Performance.csv'", type=["csv"]
)
df = load_data(uploaded_file)

# Dynamic Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filter Dataset")

selected_years = st.sidebar.multiselect(
    "Club Join Date:",
    options=sorted(df["Club_Join_Date"].unique()),
    default=sorted(df["Club_Join_Date"].unique()),
)

score_range = st.sidebar.slider(
    "Filter by Math Score Range:",
    int(df["Math_Score"].min()),
    int(df["Math_Score"].max()),
    (int(df["Math_Score"].min()), int(df["Math_Score"].max())),
)

# Apply Filters
filtered_df = df[
    (df["Club_Join_Date"].isin(selected_years))
    & (df["Math_Score"].between(score_range[0], score_range[1]))
]

# 4. Header Section
st.title("📊 Student Score Analysis & Analytics")
st.write(
    "An interactive dashboard built from **NaviBayes.ipynb** to analyze student performance metrics."
)

# 5. Key Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Students", len(filtered_df))
m2.metric("Avg Math Score", f"{filtered_df['Math_Score'].mean():.1f}")
m3.metric("Avg Reading Score", f"{filtered_df['Reading_Score'].mean():.1f}")
m4.metric("Avg Writing Score", f"{filtered_df['Writing_Score'].mean():.1f}")
m5.metric("Avg Placement Score", f"{filtered_df['Placement_Score'].mean():.1f}")

st.markdown("---")

# 6. Main Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Dataset View", "🔥 Correlation Analysis", "📈 Distributions", "🔍 Scatter Matrix"]
)

# --- TAB 1: DATASET VIEW ---
with tab1:
    st.subheader("Student Performance Data")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.dataframe(filtered_df, use_container_width=True)

    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(filtered_df.describe().T[["mean", "std", "min", "max"]])

# --- TAB 2: CORRELATION ANALYSIS ---
with tab2:
    st.subheader("Correlation Analysis")
    score_cols = ["Math_Score", "Reading_Score", "Writing_Score", "Placement_Score"]
    corr_matrix = filtered_df[score_cols].corr()

    c1, c2 = st.columns([2, 1])

    with c1:
        # Plotly Heatmap
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".4f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap of Scores",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with c2:
        st.markdown("**Correlation Matrix Values**")
        st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))


# --- TAB 3: DISTRIBUTIONS ---
with tab3:
    st.subheader("Score Distributions")
    selected_score = st.selectbox("Select Metric:", score_cols)

    d1, d2 = st.columns(2)

    with d1:
        fig_hist = px.histogram(
            filtered_df,
            x=selected_score,
            color="Club_Join_Date",
            marginal="rug",
            title=f"Distribution of {selected_score.replace('_', ' ')}",
            barmode="overlay",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with d2:
        fig_box = px.box(
            filtered_df,
            x="Club_Join_Date",
            y=selected_score,
            color="Club_Join_Date",
            title=f"{selected_score.replace('_', ' ')} by Club Join Date",
        )
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 4: SCATTER MATRIX ---
with tab4:
    st.subheader("Score Relationships Scatter Matrix")
    fig_splom = px.scatter_matrix(
        filtered_df,
        dimensions=score_cols,
        color="Club_Join_Date",
        title="Pairwise Relationships Between Scores",
    )
    fig_splom.update_traces(diagonal_visible=False)
    st.plotly_chart(fig_splom, use_container_width=True)

```

---

### Prerequisites & Setup

To run the app locally, install the required packages and execute the run command:

```bash
pip install streamlit pandas numpy plotly matplotlib seaborn
streamlit run app.py
