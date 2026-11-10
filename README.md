# GeneVision AI – Cancer Classification Using Gene-Expression Data

An educational bioinformatics and machine-learning project that classifies gene-expression samples into five tumour categories: BRCA, COAD, KIRC, LUAD and PRAD.

> **Research prototype only. This is not a clinically validated diagnostic tool and must not be used for medical decisions.**

## Dataset

UCI Machine Learning Repository: [Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq).

The project dataset used in the reported experiment contained 801 samples and 20,531 gene-expression features. The data are not included in this GitHub package. Download them from UCI and place the correctly oriented files here:

- `data/raw/data.csv`: rows are samples; columns are gene-expression features; first column contains sample IDs.
- `data/raw/labels.csv`: first column contains matching sample IDs; a `Class` column contains labels.

Do not upload patient-identifiable or restricted data to GitHub.

## Project structure

```text
GeneVision_AI_GitHub/
├── app.py
├── prediction_backend.py
├── shap_backend.py
├── train_model.py
├── requirements.txt
├── models/                  # trained artifact generated locally (not committed)
├── data/raw/                # place downloaded dataset here (not committed)
├── reports/figures/         # example evaluation figure
├── reports/tables/          # documented project result tables
└── src/
```

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

After placing `data.csv` and `labels.csv` in `data/raw/`:

```powershell
python train_model.py
```

This creates `models/final_model.joblib` and training metrics under `reports/tables/`.

## Run the application

```powershell
streamlit run app.py
```

The app expects a trained model artifact at `models/final_model.joblib`. Upload CSV samples containing the required gene-expression columns in the same feature naming convention and order as the training dataset.

## Reported experiment results

The previously documented experiment reported a Linear SVM with L1-based feature selection (50 features), evaluated on a held-out test set of 161 samples:

- Accuracy: 98.76%
- Balanced accuracy: 98.57%
- Macro precision: 99.35%
- Macro recall: 98.57%
- Macro F1: 98.93%
- Correct predictions: 159/161

These are the documented results from the earlier experiment, not a guarantee that a fresh run of the included training script will reproduce them. Exact reproduction requires the original saved fitted model, preprocessing configuration, and software environment. The training script is a reproducible implementation template based on the documented approach.

## Important interpretation notes

- This is five-class tumour-category classification, not cancer-versus-healthy detection.
- PCA is used for exploratory visualisation, not as the final classifier.
- Predictive feature importance does not prove causation or biomarker validity.
- External validation and biological verification are needed before broader claims.
- The explanation backend provides coefficient-based contributions for a linear model; do not label these values as SHAP values unless a true SHAP explainer is used.

## License

Add a license only after confirming the licensing terms of all included source code and data.
