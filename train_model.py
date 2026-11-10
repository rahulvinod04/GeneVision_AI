"""Train GeneVision AI model from the UCI-format local CSV files.

Expected files:
  data/raw/data.csv   (rows=samples, columns=gene features; first column sample ID)
  data/raw/labels.csv (first column sample ID, column named Class)
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "tables"


def main():
    X = pd.read_csv(DATA / "data.csv", index_col=0)
    labels = pd.read_csv(DATA / "labels.csv", index_col=0)
    if "Class" not in labels:
        raise ValueError("labels.csv must contain a 'Class' column")
    y = labels["Class"]
    X = X.loc[y.index]
    X.columns = X.columns.astype(str)
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Data contain missing/non-numeric values; clean before training")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    selector = SelectFromModel(
        LinearSVC(penalty="l1", loss="squared_hinge", dual=False,
                  C=0.01, max_iter=20000, random_state=42),
        max_features=50, threshold=-float("inf")
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", selector),
        ("classifier", LinearSVC(penalty="l2", C=1.0,
                                 max_iter=20000, random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_result = cross_validate(model, X_train, y_train, cv=cv,
                               scoring={"macro_f1": "f1_macro",
                                        "balanced_accuracy": "balanced_accuracy"},
                               n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "cv_macro_f1_mean": cv_result["test_macro_f1"].mean(),
        "cv_balanced_accuracy_mean": cv_result["test_balanced_accuracy"].mean(),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True)
    }
    selected_mask = model.named_steps["feature_selection"].get_support()
    selected_features = X.columns[selected_mask].tolist()
    MODELS.mkdir(exist_ok=True, parents=True)
    REPORTS.mkdir(exist_ok=True, parents=True)
    artifact = {"pipeline": model, "feature_names": X.columns.tolist(),
                "classes": list(map(str, model.classes_)),
                "selected_features": selected_features}
    joblib.dump(artifact, MODELS / "final_model.joblib")
    with open(REPORTS / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.Series(selected_features, name="feature").to_csv(REPORTS / "selected_features.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "classification_report")}, indent=2))
    print(f"Selected features: {len(selected_features)}")
    print("Saved model to models/final_model.joblib")

if __name__ == "__main__":
    main()
