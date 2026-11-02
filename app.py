"""Student Outcome Classifier - Streamlit front end.

Loads the five pre-trained pipelines from model/ and evaluates them on an
uploaded test CSV. Models are never retrained here.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics import evaluate

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}
    return meta, models


def read_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


meta, models = load_artifacts()
features, target, classes = meta["features"], meta["target"], meta["classes"]

st.title("🎓 Student Outcome Classifier")
st.caption(
    "Predicts whether a student will **Drop out**, stay **Enrolled**, or **Graduate** "
    f"using {len(features)} enrolment, academic and socio-economic features "
    "(UCI dataset id 697). Upload test data to evaluate the pre-trained models."
)

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + Target)", type=["csv"])
    sample = ROOT / "test_data.csv"
    if sample.exists():
        st.download_button("Download sample test_data.csv", sample.read_bytes(), "test_data.csv", "text/csv")
    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(models))
    st.divider()
    st.caption(f"Models trained on {meta['n_train']} rows; held-out test split has {meta['n_test']} rows.")

if uploaded is None:
    st.info("⬅️ Upload a test CSV in the sidebar to begin. You can download the sample file there.")
    with st.expander("Expected columns"):
        st.write(features + [target])
    st.stop()

data = read_upload(uploaded)
missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s): {missing}")
    st.stop()

X, y = data[features], data[target].astype(str).str.strip()
unknown = sorted(set(y) - set(classes))
if unknown:
    st.error(f"Target column has unknown labels {unknown}. Expected: {classes}")
    st.stop()

st.success(f"Loaded **{len(data)}** rows from `{uploaded.name}`.")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), width="stretch")

model = models[model_name]
y_pred = model.predict(X)
scores = evaluate(y, y_pred, model.predict_proba(X), list(model.classes_))

tab_model, tab_compare = st.tabs([f"📊 {model_name}", "🏆 Compare all models"])

with tab_model:
    items = list(scores.items())
    for row in (items[:3], items[3:]):
        for col, (k, v) in zip(st.columns(3), row):
            col.metric(k, f"{v:.4f}", border=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion(y, y_pred, classes))
    with right:
        st.subheader("Classification report")
        report = classification_report(y, y_pred, labels=classes, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    st.subheader("Predictions")
    out = data.assign(Predicted=y_pred, Correct=(y_pred == y))
    st.dataframe(out[[target, "Predicted", "Correct"] + features], width="stretch", height=280)
    st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

with tab_compare:
    rows = []
    for name, m in models.items():
        rows.append({"ML Model Name": name, **evaluate(y, m.predict(X), m.predict_proba(X), list(m.classes_))})
    table = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, props="background-color:#2e7d32;color:white;font-weight:700"), width="stretch")

    metric = st.radio("Metric to chart", list(table.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ordered = table[metric].sort_values()
    ax.barh(ordered.index, ordered.values, color=["#9ecae1"] * (len(ordered) - 1) + ["#2171b5"])
    for i, v in enumerate(ordered.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"Best on this data by MCC: **{table['MCC'].idxmax()}**")
