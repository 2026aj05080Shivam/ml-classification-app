"""Train five classifiers on the UCI Students' Dropout & Academic Success dataset.

Run:  python train.py
Outputs:
  model/<name>.joblib   - fitted sklearn pipelines (preprocessing + classifier)
  model/metadata.json   - feature list, class labels, split info
  model/metrics.csv     - evaluation metrics on the held-out test split
  test_data.csv         - the held-out test split (features + Target), for the Streamlit app
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from metrics import evaluate

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Target"
SEED = 42


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # The raw file has stray whitespace in some headers (e.g. a trailing tab).
    df.columns = [c.strip() for c in df.columns]
    return df


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
        ]),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    df = load_dataset(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    classes = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    MODEL_DIR.mkdir(exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / file_name(name))
        scores = evaluate(y_test, pipe.predict(X_test), pipe.predict_proba(X_test), list(pipe.classes_))
        rows.append({"ML Model Name": name, **scores})
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    metadata = {
        "dataset": "UCI Predict Students' Dropout and Academic Success (id 697)",
        "features": list(X.columns),
        "target": TARGET,
        "classes": classes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "models": {name: file_name(name) for name in build_models()},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
