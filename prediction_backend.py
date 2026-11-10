"""Prediction utilities for the GeneVision AI Streamlit application."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.joblib"


def load_model_artifact(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Train the model first using `python train_model.py`."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"pipeline", "feature_names", "classes"}.issubset(artifact):
        raise ValueError("Model artifact must contain pipeline, feature_names and classes.")
    return artifact


def prepare_input(uploaded_df, feature_names):
    if uploaded_df is None or uploaded_df.empty:
        raise ValueError("The uploaded CSV contains no samples.")
    df = uploaded_df.copy()
    df.columns = df.columns.astype(str).str.strip()
    id_cols = [c for c in df.columns if c.lower() in {
        "sample", "sample_id", "sampleid", "id", "patient_id", "patientid", "unnamed: 0"
    }]
    df = df.drop(columns=id_cols, errors="ignore")
    required = list(map(str, feature_names))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required gene features. Examples: {missing[:8]}")
    X = df.loc[:, required].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Gene-expression columns contain missing or non-numeric values.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Gene-expression data contain infinite values.")
    return X


def predict_samples(uploaded_df, artifact=None):
    artifact = artifact or load_model_artifact()
    X = prepare_input(uploaded_df, artifact["feature_names"])
    pipeline = artifact["pipeline"]
    pred = pipeline.predict(X)
    out = pd.DataFrame({"Predicted Cancer Type": pred})
    # LinearSVC decision scores are not probabilities; do not label them as probabilities.
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(X)
            for i, cls in enumerate(pipeline.classes_):
                out[f"{cls} Probability (%)"] = probs[:, i] * 100
        except (AttributeError, NotImplementedError):
            pass
    return out
