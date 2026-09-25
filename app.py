from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Student Performance | KNN Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #16211f; --muted: #66736e; --mint: #b9e7d4; --lime: #d9f27c; --paper: #f5f7f2; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h1 { font-size: clamp(2.2rem, 5vw, 4.4rem); line-height: .98; margin-bottom: .5rem; }
    .hero { background: #16211f; color: #f5f7f2; padding: 2.2rem 2.4rem; border-radius: 18px; margin: .5rem 0 1.5rem; }
    .hero .eyebrow { color: var(--lime); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .hero p { color: #c9d5d0; max-width: 760px; font-size: 1.05rem; margin: 0; }
    .metric-card { background: white; border: 1px solid #dfe7e1; border-radius: 12px; padding: 1rem 1.1rem; min-height: 98px; }
    .metric-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; margin-top: .3rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div[data-testid="stSidebar"] { background: #e4eee8; }
    .note { background: #edf6d1; border-left: 4px solid #9bbb42; padding: .8rem 1rem; border-radius: 6px; color: #34421e; }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_FILE = Path(__file__).with_name("Students Performance.csv")
REQUIRED_COLUMNS = ["Math_Score", "Reading_Score", "Writing_Score", "Placement_Score", "Club_Join_Date"]
SCORE_COLUMNS = REQUIRED_COLUMNS[:4]
BAND_ORDER = ["Low", "Medium", "High"]
BAND_COLORS = {"Low": "#e86f51", "Medium": "#f2c14e", "High": "#46a884"}


def load_data(uploaded_file):
    source = uploaded_file if uploaded_file is not None else DEFAULT_FILE
    data = pd.read_csv(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    if data.empty:
        raise ValueError("The dataset has no complete numeric rows after cleaning.")
    data["Total_Score"] = data[SCORE_COLUMNS].sum(axis=1)
    data["Performance_Band"] = pd.cut(
        data["Total_Score"],
        bins=[-np.inf, 280, 361, np.inf],
        labels=BAND_ORDER,
        right=False,
    ).astype(str)
    return data


def make_model(k):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]
    )


def fit_knn(data, features, k, test_size, random_state):
    X = data[features]
    y = data["Performance_Band"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = make_model(k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, predictions


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><div class="eyebrow">Student score intelligence / KNN studio</div>'
    '<h1>Find the students<br>closest in performance.</h1>'
    '<p>An interactive K-nearest neighbors lab for exploring score patterns, model behavior, '
    'and individual performance predictions.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model controls")
    uploaded_file = st.file_uploader("Use another CSV", type=["csv"])
    st.caption("Required columns: Math_Score, Reading_Score, Writing_Score, Placement_Score, Club_Join_Date")
    st.divider()
    k_value = st.slider("Neighbors (k)", min_value=1, max_value=31, value=7, step=2)
    test_size = st.slider("Test set size", min_value=0.15, max_value=0.40, value=0.20, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=999, value=42, step=1)

try:
    data = load_data(uploaded_file)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    default_features = SCORE_COLUMNS + ["Club_Join_Date"]
    features = st.multiselect(
        "Predictor features",
        REQUIRED_COLUMNS,
        default=default_features,
        help="Features are standardized before distance calculations. Select at least two.",
    )
    if len(features) < 2:
        st.warning("Select at least two predictor features.")
        st.stop()

model, X_train, X_test, y_train, y_test, predictions = fit_knn(
    data, features, k_value, test_size, int(random_state)
)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, labels=BAND_ORDER, output_dict=True, zero_division=0)

metric_columns = st.columns(4)
with metric_columns[0]:
    metric_card("Rows analysed", f"{len(data):,}")
with metric_columns[1]:
    metric_card("Model accuracy", f"{accuracy:.1%}")
with metric_columns[2]:
    metric_card("Training rows", f"{len(X_train):,}")
with metric_columns[3]:
    metric_card("Active features", str(len(features)))

st.markdown("<br>", unsafe_allow_html=True)
tab_overview, tab_model, tab_predict, tab_data = st.tabs(
    ["Overview", "Model diagnostics", "Predict a student", "Dataset"]
)

with tab_overview:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Performance landscape")
        scatter_features = st.multiselect(
            "Choose the chart axes",
            SCORE_COLUMNS,
            default=["Math_Score", "Placement_Score"],
            max_selections=2,
            key="scatter_features",
        )
        if len(scatter_features) == 2:
            chart = px.scatter(
                data,
                x=scatter_features[0],
                y=scatter_features[1],
                color="Performance_Band",
                category_orders={"Performance_Band": BAND_ORDER},
                color_discrete_map=BAND_COLORS,
                hover_data=["Total_Score", "Club_Join_Date"],
                template="simple_white",
            )
            chart.update_layout(height=430, legend_title_text="Band", margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("Band composition")
        counts = data["Performance_Band"].value_counts().reindex(BAND_ORDER).fillna(0).astype(int)
        bar = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=BAND_COLORS,
            labels={"x": "Performance band", "y": "Students"},
            template="simple_white",
        )
        bar.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(bar, use_container_width=True)
        st.markdown(
            '<div class="note"><strong>How labels are created:</strong> the notebook’s four scores '
            'are summed into <em>Total_Score</em>. Low is below 280, Medium is 280–360, and High is above 360.</div>',
            unsafe_allow_html=True,
        )

with tab_model:
    st.subheader("What is the model seeing?")
    col_a, col_b = st.columns([1, 1.25])
    with col_a:
        st.write(f"KNN uses **{k_value}** distance-weighted neighbors after standardization.")
        st.dataframe(
            pd.DataFrame(
                {
                    "Band": BAND_ORDER,
                    "Precision": [report[label]["precision"] for label in BAND_ORDER],
                    "Recall": [report[label]["recall"] for label in BAND_ORDER],
                    "F1": [report[label]["f1-score"] for label in BAND_ORDER],
                }
            ).style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
    with col_b:
        matrix = confusion_matrix(y_test, predictions, labels=BAND_ORDER)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[f"Predicted {label}" for label in BAND_ORDER],
                y=[f"Actual {label}" for label in BAND_ORDER],
                colorscale=[[0, "#f5f7f2"], [1, "#46a884"]],
                text=matrix,
                texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
            )
        )
        heatmap.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(heatmap, use_container_width=True)

with tab_predict:
    st.subheader("Estimate one student's performance band")
    st.caption("Adjust the inputs, then select Predict. The model compares the student with the training set.")
    input_columns = st.columns(2)
    student_values = {}
    for index, feature in enumerate(features):
        with input_columns[index % 2]:
            minimum = int(data[feature].min())
            maximum = int(data[feature].max())
            student_values[feature] = st.number_input(
                feature.replace("_", " "),
                min_value=minimum,
                max_value=maximum,
                value=int(data[feature].median()),
                step=1,
                key=f"input_{feature}",
            )
    if st.button("Predict performance band", type="primary", use_container_width=True):
        student = pd.DataFrame([student_values], columns=features)
        prediction = model.predict(student)[0]
        probabilities = model.predict_proba(student)[0]
        classes = model.named_steps["knn"].classes_
        st.success(f"Predicted band: {prediction}")
        probability_chart = px.bar(
            x=classes,
            y=probabilities,
            color=classes,
            color_discrete_map=BAND_COLORS,
            labels={"x": "Performance band", "y": "Probability"},
            template="simple_white",
        )
        probability_chart.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(probability_chart, use_container_width=True)

with tab_data:
    st.subheader("Prepared dataset")
    st.dataframe(data, use_container_width=True, height=430)
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download prepared CSV", csv, "students_performance_prepared.csv", "text/csv")
