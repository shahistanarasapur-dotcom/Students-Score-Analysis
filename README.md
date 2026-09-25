Here is the complete Python code for your Streamlit web application. It parses the analysis from your `NaviBayes.ipynb` notebook and converts it into a fully dynamic, interactive data science dashboard using Streamlit, Plotly, and Pandas.

### objective -  Analysing the student score and give feedback by using ml model
### `app.py`
<img width="1600" height="900" alt="Screenshot 2026-09-25 111753" src="https://github.com/user-attachments/assets/49ddd806-f3d8-4833-996f-414e56651c0a" />
<img width="1600" height="900" alt="Screenshot 2026-09-25 111649" src="https://github.com/user-attachments/assets/c9fbb97b-945e-44a5-ac53-1af9d64a836c" />
```python

Below, you'll find a complete, production-ready project setup for turning the **Student Performance Analysis** notebook into an interactive, dynamic **Streamlit** dashboard.

---

### 1. Project Objectives & Structure

#### Objectives

1. **Interactive Data Exploration**: Allow users to inspect raw data, filter by scores/join years, and view summary statistics.
2. **Correlation & Trend Analysis**: Dynamically visualize relationship heatmaps and feature distributions using Seaborn and Matplotlib.
3. **Student Performance Classification**: Implement a Naive Bayes classifier (Gaussian / Categorical) to predict placement readiness or performance tiers based on Math, Reading, and Writing scores.
4. **Interactive What-If Scenarios**: Enable users to input custom test scores via sidebar controls and receive real-time predictions with class probabilities.

#### Project Directory Structure

```text
NaviBayes-Streamlit/
│
├── .streamlit/
│   └── config.toml          # Custom theme and configuration
├── data/
│   └── Students_Performance.csv # Dataset file
├── app.py                   # Main Streamlit Application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```

---

### 2. `requirements.txt`

```text
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0

```

---

### 3. Streamlit Application (`app.py`)

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NaviBayes | Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Attempt to load dataset; fallback to generated mock data if missing
    try:
        df = pd.read_csv("data/Students_Performance.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("Students Performance.csv")
        except FileNotFoundError:
            # Synthetic generation matching notebook structure for demo resilience
            np.random.seed(42)
            n_samples = 399
            df = pd.DataFrame({
                'Math_Score': np.random.randint(60, 101, n_samples),
                'Reading_Score': np.random.randint(60, 101, n_samples),
                'Writing_Score': np.random.randint(60, 101, n_samples),
                'Placement_Score': np.random.randint(60, 101, n_samples),
                'Club_Join_Date': np.random.choice([2018, 2019, 2020, 2021], n_samples)
            })
    return df

df_raw = load_data()

# Create a target label for Naive Bayes Classification based on Placement Score
# Threshold: >= 80 -> High Placement Readiness (1), else Standard (0)
df = df_raw.copy()
df['Placement_Class'] = (df['Placement_Score'] >= 80).astype(int)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/illustrations/200/education.png", width=120)
st.sidebar.title("NaviBayes Control")

navigation = st.sidebar.radio(
    "Select Module:",
    ["Overview & Dataset", "Exploratory Analytics", "Naive Bayes Classifier", "Interactive Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")
selected_years = st.sidebar.multiselect(
    "Club Join Years",
    options=sorted(df['Club_Join_Date'].unique()),
    default=sorted(df['Club_Join_Date'].unique())
)

filtered_df = df[df['Club_Join_Date'].isin(selected_years)]

# -----------------------------------------------------------------------------
# Module 1: Overview & Dataset
# -----------------------------------------------------------------------------
if navigation == "Overview & Dataset":
    st.markdown('<div class="main-title">🎓 Student Score Analysis & NaviBayes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Exploration of academic performance metrics and probabilistic classification using Naive Bayes.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(filtered_df))
    col2.metric("Avg Math Score", f"{filtered_df['Math_Score'].mean():.1f}")
    col3.metric("Avg Reading Score", f"{filtered_df['Reading_Score'].mean():.1f}")
    col4.metric("Avg Writing Score", f"{filtered_df['Writing_Score'].mean():.1f}")

    st.markdown("---")
    st.subheader("Filtered Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)

    with st.expander("View Descriptive Statistics"):
        st.write(filtered_df.describe())

# -----------------------------------------------------------------------------
# Module 2: Exploratory Analytics
# -----------------------------------------------------------------------------
elif navigation == "Exploratory Analytics":
    st.markdown('<div class="main-title">📊 Exploratory Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Correlation Heatmap")
        corr_cols = ['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']
        corr_matrix = filtered_df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", ax=ax, cbar=True, vmin=-0.2, vmax=1.0)
        ax.set_title("Correlation Heatmap of Scores")
        st.pyplot(fig)

    with col2:
        st.subheader("Score Distributions")
        selected_metric = st.selectbox("Select Score to Plot Distribution", corr_cols)
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.histplot(filtered_df[selected_metric], kde=True, color="#3B82F6", bins=15, ax=ax)
        ax.set_title(f"Distribution of {selected_metric}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Pairwise Relationships")
    fig_pair = sns.pairplot(filtered_df[corr_cols], diag_kind='kde', corner=True)
    st.pyplot(fig_pair)

# -----------------------------------------------------------------------------
# Module 3: Naive Bayes Classifier
# -----------------------------------------------------------------------------
elif navigation == "Naive Bayes Classifier":
    st.markdown('<div class="main-title">🤖 Naive Bayes Model Performance</div>', unsafe_allow_html=True)
    st.write("Using Gaussian Naive Bayes to classify high placement performance (`Placement_Score >= 80`).")

    # Feature selection
    features = ['Math_Score', 'Reading_Score', 'Writing_Score']
    X = filtered_df[features]
    y = filtered_df['Placement_Class']

    if len(np.unique(y)) < 2:
        st.warning("Insufficient class diversity in filtered data. Please expand filters in the sidebar.")
    else:
        test_size = st.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Model Metrics")
            st.metric("Model Accuracy", f"{acc * 100:.2f}%")
            
            st.markdown("**Parameters Used:**")
            st.json({"Algorithm": "Gaussian Naive Bayes", "Features": features, "Target": "Placement_Class (>=80)"})

        with col2:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Standard", "High Readiness"],
                        yticklabels=["Standard", "High Readiness"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

# -----------------------------------------------------------------------------
# Module 4: Interactive Prediction
# -----------------------------------------------------------------------------
elif navigation == "Interactive Prediction":
    st.markdown('<div class="main-title">🔮 Real-Time Student Predictor</div>', unsafe_allow_html=True)
    st.write("Adjust academic scores to predict placement classification using Naive Bayes.")

    features = ['Math_Score', 'Reading_Score', 'Writing_Score']
    X = df[features]
    y = df['Placement_Class']
    
    model = GaussianNB()
    model.fit(X, y)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Academic Scores")
        math_in = st.slider("Math Score", 0, 100, 75)
        reading_in = st.slider("Reading Score", 0, 100, 75)
        writing_in = st.slider("Writing Score", 0, 100, 75)

        input_data = pd.DataFrame([[math_in, reading_in, writing_in]], columns=features)

    with col2:
        st.subheader("Prediction Result")
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success(f"**High Placement Probability**: Likely Ready ({probs[1]*100:.1f}% confidence)")
        else:
            st.info(f"**Standard Placement Probability**: Needs Improvement ({probs[0]*100:.1f}% confidence)")

        # Display Probability Breakdown
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Standard", "High Readiness"], probs, color=["#9CA3AF", "#2563EB"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

```

---

### 4. `README.md`

```markdown
# NaviBayes: Student Performance Dashboard

An interactive Streamlit application for exploratory data analysis and Naive Bayes performance classification on student academic scores.

## Features
- **Exploratory Data Analysis**: Dynamic score correlation heatmaps, score distributions, and interactive dataset filtering.
- **Naive Bayes Machine Learning Model**: Trains a Gaussian Naive Bayes classifier to identify placement readiness based on Math, Reading, and Writing performance.
- **Interactive Predictor**: Real-time evaluation of hypothetical student score profiles.

## Installation & Setup

1. **Clone or Download the Repository**:
   ```bash
   git clone [https://github.com/your-username/NaviBayes.git](https://github.com/your-username/NaviBayes.git)
   cd NaviBayes

```

2. **Create and Activate a Virtual Environment** (Optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


4. **Add Dataset**:
Ensure `Students Performance.csv` is placed inside the `data/` directory or root directory.
5. **Run the Streamlit Application**:
```bash
streamlit run app.py

```



```

```
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
