"""SHAP-style linear explanation for the fitted GeneVision Linear SVM.

For a linear decision function and a zero-valued reference in the scaled,
selected-feature space, feature contributions are coef * transformed_value.
This is a model explanation under that explicit baseline, not a biological
causality claim. It is not a KernelExplainer/TreeExplainer computation.
"""
import numpy as np
import pandas as pd
from prediction_backend import load_model_artifact, prepare_input


def explain_sample(sample_df, artifact=None, top_n=20):
    artifact = artifact or load_model_artifact()
    X = prepare_input(sample_df, artifact["feature_names"])
    pipeline = artifact["pipeline"]
    predicted = str(pipeline.predict(X)[0])

    scaler = pipeline.named_steps.get("scaler")
    selector = pipeline.named_steps.get("feature_selection")
    classifier = pipeline.named_steps.get("classifier")
    transformed = X
    if scaler is not None:
        transformed = scaler.transform(transformed)
    feature_names = np.asarray(artifact["feature_names"])
    if selector is not None:
        transformed = selector.transform(transformed)
        if hasattr(selector, "get_support"):
            feature_names = feature_names[selector.get_support()]
    values = np.asarray(transformed)[0]

    contributions = np.zeros(len(feature_names), dtype=float)
    if classifier is not None and hasattr(classifier, "coef_"):
        classes = list(getattr(classifier, "classes_", artifact.get("classes", [])))
        class_index = classes.index(predicted) if predicted in classes else 0
        # LinearSVC multiclass uses one coefficient row per class.
        coef = classifier.coef_[min(class_index, len(classifier.coef_) - 1)]
        contributions = np.asarray(coef) * values

    table = pd.DataFrame({
        "Feature": feature_names.astype(str),
        "Original Feature": feature_names.astype(str),
        "Gene Symbol": "Not annotated",
        "Entrez ID": "Not mapped",
        "SHAP Value": contributions,
        "Absolute SHAP": np.abs(contributions),
    }).sort_values("Absolute SHAP", ascending=False).head(top_n).reset_index(drop=True)
    return predicted, table
