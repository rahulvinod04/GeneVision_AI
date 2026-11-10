import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from prediction_backend import predict_samples
from shap_backend import explain_sample

st.set_page_config(
    page_title="GeneVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DESIGN SYSTEM --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root { --navy:#07111f; --navy2:#0d1b2d; --teal:#42e8d5; --teal2:#14b8a6; --ink:#eaf2fb; --muted:#9aadc2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(20,184,166,.13), transparent 32%),
    radial-gradient(ellipse at 90% 12%, rgba(59,130,246,.12), transparent 30%),
    linear-gradient(180deg, #07111f 0%, #0a1727 52%, #0b1422 100%);
  color:var(--ink);
}
.block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1c2e,#112b43); border-right:1px solid #203b53; }
[data-testid="stSidebar"] * { color:#e8f0f8; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aebfd1; }
[data-testid="stSidebar"] hr { border-color:#29445d; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:8px; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:#edf5ff!important; letter-spacing:-.025em; }
h1 { font-size:2.15rem!important; }
.hero {
  position:relative; overflow:hidden; padding:34px 38px; border-radius:22px;
  background:linear-gradient(120deg,#0b1c2e 0%,#123b51 55%,#087f82 100%);
  box-shadow:0 14px 35px rgba(15,45,68,.16); margin-bottom:22px;
}
.hero:after { content:""; position:absolute; right:-55px; top:-110px; width:330px; height:330px;
  border:1px solid rgba(255,255,255,.13); border-radius:50%; box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 75px rgba(255,255,255,.025); }
.hero-kicker { color:#7ee9db; font-size:11px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-family:'Manrope',sans-serif; color:white; font-size:34px; font-weight:800; line-height:1.2; margin:9px 0; }
.hero-sub { color:#d4e7f0; font-size:15px; max-width:700px; }
.pill { display:inline-block; background:rgba(126,233,219,.12); border:1px solid rgba(126,233,219,.28); color:#a9f4e9; padding:5px 10px; border-radius:30px; font-size:11px; font-weight:700; }
.panel { background:rgba(17,31,49,.76); border:1px solid rgba(148,163,184,.18); border-radius:18px; padding:22px; box-shadow:0 12px 32px rgba(0,0,0,.20); backdrop-filter:blur(18px); }
.feature { background:linear-gradient(145deg,rgba(19,35,55,.92),rgba(12,24,39,.88)); border:1px solid rgba(148,163,184,.18); border-radius:17px; padding:20px; height:100%; transition:transform .15s ease, box-shadow .15s ease; box-shadow:0 10px 28px rgba(0,0,0,.16); backdrop-filter:blur(16px); }
.feature:hover { transform:translateY(-3px); box-shadow:0 16px 34px rgba(0,0,0,.25); border-color:rgba(66,232,213,.42); }
.feature-icon { width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:12px;background:#e7f8f5;font-size:22px;margin-bottom:12px; }
.feature-title { font-weight:700;color:#edf5ff;font-size:16px;margin-bottom:6px; }
.feature-text { color:#a7b8cc;font-size:13px;line-height:1.5; }
.stat { background:rgba(17,31,49,.82); border:1px solid rgba(148,163,184,.18); border-radius:15px; padding:16px 18px; backdrop-filter:blur(16px); }
.stat-label { color:#9aadc2; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.stat-value { color:#f0f7ff; font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;margin-top:5px; }
.stat-desc { color:#0d9488;font-size:12px;margin-top:2px; }
div.stButton>button { border-radius:10px; border:1px solid #0d9488; background:#0d9488; color:white; font-weight:700; padding:.6rem 1rem; transition:all .15s ease; }
div.stButton>button:hover { background:#087f76; border-color:#087f76; color:white; transform:translateY(-1px); }
div[data-testid="stFileUploader"] { background:rgba(15,29,47,.78);border:1px dashed rgba(66,232,213,.38);border-radius:14px;padding:12px; }
div[data-testid="stTabs"] button { font-weight:600; }
[data-testid="stMetric"] { background:rgba(17,31,49,.82);border:1px solid rgba(148,163,184,.18);border-radius:14px;padding:14px; }
.footer { color:#7f93aa;font-size:11px;text-align:center;padding:18px 0 0; }

/* Premium dark theme surfaces and text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
.stText, .stCaption, label {
  color: #d8e5f3;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: rgba(12, 24, 39, .92);
}
input, textarea, [data-baseweb="select"] > div {
  background: rgba(13, 27, 44, .92) !important;
  color: #eaf2fb !important;
  border-color: rgba(148,163,184,.25) !important;
}
[data-testid="stTabs"] button {
  color: #c7d5e5 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #42e8d5 !important;
  border-bottom-color: #42e8d5 !important;
}
div[data-testid="stAlert"] {
  background: rgba(17,31,49,.88);
  border: 1px solid rgba(148,163,184,.18);
}


.top-brand-name { color:#f4f8fc !important; }
.top-brand-sub { color:#9aadc2 !important; }

</style>
""", unsafe_allow_html=True)

# -------------------- STATE / NAV --------------------
if "gv_page" not in st.session_state:
    st.session_state.gv_page = "⌂  Overview"
if "gv_file" not in st.session_state:
    st.session_state.gv_file = None
if "gv_history" not in st.session_state:
    st.session_state.gv_history = []

def go(page):
    st.session_state.gv_page = page
    if "gv_top_nav" in st.session_state:
        st.session_state.gv_top_nav = page

# Horizontal top navigation
pages = ["⌂  Overview", "🧬  Analyze Sample", "⌕  Sample Explorer", "◷  History", "ⓘ  About"]
st.markdown("""
<style>
/* Hide the Streamlit sidebar completely */
[data-testid="stSidebar"] {display:none;}
[data-testid="collapsedControl"] {display:none;}
.top-brand {
  display:flex; align-items:center; justify-content:space-between;
  padding: 4px 0 18px; margin-bottom: 10px;
}
.top-brand-name {font-family:Manrope,sans-serif;font-weight:800;font-size:42px;line-height:1.12;color:#f4f8fc !important;letter-spacing:-.04em;text-shadow:0 2px 18px rgba(255,255,255,.06);}
.top-brand-sub {font-size:11px;letter-spacing:.16em;font-weight:700;color:#9aadc2;margin-top:8px;}
/* iOS-inspired neutral WHITE glass navigation */
div[data-testid="stRadio"] {
  position: sticky;
  top: 0;
  z-index: 999;
  width: 100%;
  margin: 12px 0 30px 0;
  padding: 12px 0 17px;
  background: rgba(8, 17, 31, 0.62);
  backdrop-filter: saturate(150%) blur(26px);
  -webkit-backdrop-filter: saturate(150%) blur(26px);
  border-bottom: 1px solid rgba(255,255,255,.10);
  box-shadow: 0 12px 28px rgba(0,0,0,.18);
}

/* Main frosted-white capsule: neutral white, no teal/green tint */
div[data-testid="stRadio"] > div[role="radiogroup"] {
  box-sizing: border-box;
  width: 100%;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: nowrap;
  padding: 10px;
  background: rgba(255, 255, 255, 0.17);
  backdrop-filter: saturate(135%) blur(28px);
  -webkit-backdrop-filter: saturate(135%) blur(28px);
  border: 1px solid rgba(255, 255, 255, 0.58);
  border-radius: 28px;
  box-shadow:
    0 12px 32px rgba(0,0,0,.22),
    inset 0 1px 0 rgba(255,255,255,.70),
    inset 0 -1px 0 rgba(255,255,255,.10);
}

/* Individual buttons: translucent white glass */
div[data-testid="stRadio"] > div[role="radiogroup"] label {
  flex: 1 1 0;
  min-width: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  background: rgba(255, 255, 255, 0.08) !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  border-radius: 20px !important;
  padding: 13px 12px !important;
  margin: 0 !important;
  transition: all .22s ease;
}

/* Light text with strong contrast against the dark app */
div[data-testid="stRadio"] > div[role="radiogroup"] label p {
  color: #f4f7fb !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  text-align: center;
  white-space: nowrap;
}

/* Hover is brighter neutral glass */
div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
  background: rgba(255,255,255,.23) !important;
  border-color: rgba(255,255,255,.68) !important;
  box-shadow: 0 4px 15px rgba(255,255,255,.06);
  transform: translateY(-1px);
}

/* Selected tab: white translucent capsule, not green */
div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
  background: rgba(255,255,255,.31) !important;
  border: 1px solid rgba(255,255,255,.82) !important;
  box-shadow:
    0 4px 18px rgba(255,255,255,.10),
    inset 0 1px 0 rgba(255,255,255,.82);
}

/* Hide Streamlit radio controls so no black dots appear */
div[data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child,
div[data-testid="stRadio"] > div[role="radiogroup"] label input {
  display: none !important;
}

/* Keep a visible gap between the nav and page content */
div[data-testid="stRadio"] + div {
  margin-top: 14px;
}

@media (max-width: 850px) {
  div[data-testid="stRadio"] > div[role="radiogroup"] {
    flex-wrap: wrap;
  }
  div[data-testid="stRadio"] > div[role="radiogroup"] label {
    flex: 1 1 30%;
  }
}
</style>
""", unsafe_allow_html=True)

brand_left, brand_right = st.columns([3,1])
with brand_left:
    st.markdown("""
    <div class="top-brand">
      <div>
        <div class="top-brand-name">🧬 GeneVision AI</div>
        <div class="top-brand-sub">GENE EXPRESSION • MACHINE LEARNING • EXPLAINABLE AI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
with brand_right:
    st.markdown('<div style="text-align:right;padding-top:9px"><span class="pill">● Local research workspace</span></div>', unsafe_allow_html=True)

if "gv_top_nav" not in st.session_state:
    st.session_state.gv_top_nav = st.session_state.gv_page

chosen_page = st.radio(
    "Main navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="gv_top_nav"
)
st.session_state.gv_page = chosen_page
st.markdown(
    "<hr style='border:0;border-top:1px solid #dce6ef;margin:8px 0 24px'>",
    unsafe_allow_html=True
)

# -------------------- HERO --------------------
def hero(kicker,title,subtitle):
    st.markdown(f"""
    <div class="hero">
      <div class="hero-kicker">{kicker}</div>
      <div class="hero-title">{title}</div>
      <div class="hero-sub">{subtitle}</div>
    </div>
    """,unsafe_allow_html=True)

def feature(icon,title,desc):
    st.markdown(f'<div class="feature"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>',unsafe_allow_html=True)

# -------------------- OVERVIEW --------------------
if st.session_state.gv_page=="⌂  Overview":
    hero("GENE EXPRESSION ANALYSIS","A clearer view into gene expression.","Upload a sample, explore its structure, and prepare it for explainable machine-learning analysis.")
    top1,top2=st.columns([1.5,1])
    with top1:
        st.markdown("### Welcome to GeneVision")
        st.markdown("A focused research workspace for **gene expression classification** and **model interpretability**.")
        if st.button("＋  Start sample analysis",use_container_width=False):
            go("🧬  Analyze Sample")
            st.rerun()
    with top2:
        st.markdown('<div class="panel"><span class="pill">WORKFLOW</span><h3 style="margin-top:12px">From data to insight</h3><p style="color:#718096;font-size:13px">01 Upload sample → 02 Validate format → 03 Predict class → 04 Explain with SHAP</p></div>',unsafe_allow_html=True)
    st.write("")
    c1,c2,c3=st.columns(3)
    with c1: feature("📁","Sample upload","Import a CSV file and inspect its sample and feature dimensions.")
    with c2: feature("🧬","Model prediction","Run the trained classifier on compatible gene expression input.")
    with c3: feature("✦","Explainable AI","Understand feature contributions with SHAP-based explanations.")
    st.write("")
    st.markdown("### Quick actions")
    qa1,qa2,qa3=st.columns(3)
    with qa1:
        if st.button("Upload a sample →",key="quick_upload",use_container_width=True): go("🧬  Analyze Sample"); st.rerun()
    with qa2:
        if st.button("Explore data →",key="quick_explore",use_container_width=True): go("Sample Explorer"); st.rerun()
    with qa3:
        if st.button("View history →",key="quick_history",use_container_width=True): go("History"); st.rerun()
    st.markdown('<div class="footer">GeneVision is an academic research prototype. It is not intended for clinical diagnosis.</div>',unsafe_allow_html=True)

# -------------------- ANALYZE SAMPLE --------------------
elif st.session_state.gv_page=="🧬  Analyze Sample":
    hero("ANALYSIS WORKSPACE","Analyze a gene expression sample","Upload expression data and verify the input before running the model.")
    st.markdown("### 1 · Upload data")
    upload=st.file_uploader("Drop your CSV file here or browse",type=["csv"],help="Use a CSV containing sample rows and gene-expression features.")
    if upload:
        try:
            df=pd.read_csv(upload)
            st.session_state.gv_file=upload.name
            st.session_state.gv_data=df
            st.success(f"Uploaded: {upload.name}")
            m1,m2,m3=st.columns(3)
            m1.metric("Rows / samples",f"{df.shape[0]:,}")
            m2.metric("Columns",f"{df.shape[1]:,}")
            m3.metric("Missing values",f"{int(df.isna().sum().sum()):,}")
            st.markdown("### 2 · Select sample")
            if len(df)>1:
                row_idx=st.selectbox("Choose sample row",range(len(df)),format_func=lambda i:f"Sample {i+1} · row {i}")
            else: row_idx=0
            with st.expander("Preview selected sample",expanded=True):
                st.dataframe(df.iloc[[row_idx]].T.rename(columns={row_idx:"Value"}).head(30),use_container_width=True)
                st.caption("Showing up to 30 fields. Check the model's required feature format before prediction.")
            st.markdown("### 3 · Analysis")
            st.info(
                "The trained model requires exactly the expected gene-expression features "
                "(gene_0 through gene_20530). Extra columns such as sample IDs are ignored."
            )

            if st.button("✦  Analyze selected sample", use_container_width=True):
                try:
                    selected_sample = df.iloc[[row_idx]].copy()
                    prediction_result = predict_samples(selected_sample)

                    predicted_class = prediction_result.loc[0, "Predicted Cancer Type"]
                    st.session_state.gv_prediction = prediction_result
                    st.session_state.gv_history.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "File": upload.name,
                        "Sample": f"Sample {row_idx + 1}",
                        "Prediction": predicted_class
                    })

                    st.markdown("### 4 · Prediction result")
                    st.success(f"Predicted class: **{predicted_class}**")

                    probability_columns = [
                        col for col in prediction_result.columns
                        if col.endswith("Probability (%)")
                    ]
                    if probability_columns:
                        probability_data = prediction_result[probability_columns].iloc[0]
                        probability_data.index = [
                            col.replace(" Probability (%)", "")
                            for col in probability_columns
                        ]
                        st.markdown("#### Model output scores")
                        st.bar_chart(probability_data)
                        st.dataframe(
                            probability_data.rename("Model score (%)").sort_values(ascending=False),
                            use_container_width=True
                        )
                    st.caption(
                        "Research prototype only. Model scores are not clinical probabilities "
                        "and must not be used for diagnosis or treatment."
                    )

                    st.markdown("### 5 · SHAP explainability")
                    with st.spinner("Calculating SHAP explanation for this sample…"):
                        shap_prediction, shap_table = explain_sample(selected_sample)

                    st.info(
                        f"SHAP explanation for the **{shap_prediction}** model score. "
                        "Positive values push the score upward; negative values push it downward."
                    )
                    chart_data = shap_table.set_index("Feature")["SHAP Value"]
                    st.markdown("#### Top contributing selected features")
                    st.bar_chart(chart_data)

                    st.dataframe(
                        shap_table[
                            [
                                "Feature",
                                "Original Feature",
                                "Gene Symbol",
                                "Entrez ID",
                                "SHAP Value",
                                "Absolute SHAP",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Gene annotations use a source-order-supported mapping "
                        "from the original Synapse gene list. Exact biological "
                        "alignment has not been independently validated. "
                        "SHAP explains model behavior, not biological causation."
                    )

                except Exception as e:
                    st.error(f"Prediction could not be completed: {e}")
                    st.warning(
                        "Please verify that the CSV has the required gene_0 to gene_20530 "
                        "columns and numeric expression values."
                    )
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.markdown('<div class="panel"><h4>Ready when you are</h4><p style="color:#718096">Choose a CSV file to begin. For the first integration test, we will use a single sample with the same feature structure as the training dataset.</p></div>',unsafe_allow_html=True)
    st.caption("Predictions will be research outputs only—not medical diagnoses.")

# -------------------- EXPLORER --------------------
elif st.session_state.gv_page=="⌕  Sample Explorer":
    hero("DATA EXPLORER","Inspect sample structure","Explore column names, missing values, data types, and numeric distributions.")
    data=st.session_state.get("gv_data")
    if data is None:
        st.info("Upload a CSV first in Analyze Sample.")
        if st.button("Go to upload"): go("🧬  Analyze Sample"); st.rerun()
    else:
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Samples",len(data)); d2.metric("Features",data.shape[1]); d3.metric("Null cells",int(data.isna().sum().sum())); d4.metric("Duplicate rows",int(data.duplicated().sum()))
        t1,t2,t3=st.tabs(["Data preview","Column profile","Missingness"])
        with t1:
            search=st.text_input("Search columns",placeholder="Type a column/gene name…")
            view=data
            if search: view=view.loc[:,view.columns.astype(str).str.contains(search,case=False,regex=False)]
            st.dataframe(view.head(50),use_container_width=True)
            st.download_button("Download uploaded CSV",data.to_csv(index=False),"gene_expression_uploaded.csv","text/csv")
        with t2:
            col=st.selectbox("Select feature",data.columns)
            st.write("Type:",str(data[col].dtype)," | Unique values:",int(data[col].nunique()))
            if pd.api.types.is_numeric_dtype(data[col]):
                st.write(data[col].describe())
                st.bar_chart(data[col].value_counts().sort_index().head(50))
            else:
                st.dataframe(data[col].value_counts().head(25))
        with t3:
            miss=(data.isna().sum().sort_values(ascending=False))
            miss=miss[miss>0]
            if len(miss)==0: st.success("No missing values found.")
            else: st.dataframe(miss.rename("Missing values"),use_container_width=True)

# -------------------- HISTORY --------------------
elif st.session_state.gv_page=="◷  History":
    hero("SESSION HISTORY","Recent analysis activity","Review actions performed during this current app session.")
    if not st.session_state.gv_history:
        st.markdown('<div class="panel"><h4>No analyses yet</h4><p style="color:#718096">Your analysis history will appear here after model predictions are connected.</p></div>',unsafe_allow_html=True)
    else:
        st.dataframe(pd.DataFrame(st.session_state.gv_history),use_container_width=True)

# -------------------- ABOUT --------------------
elif st.session_state.gv_page=="ⓘ  About":
    hero("PROJECT INFORMATION","About GeneVision AI","A student research project exploring machine learning and explainability in gene expression data.")
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="panel"><h3>Project overview</h3><p style="color:#718096">GeneVision aims to classify gene expression samples into five cancer categories and provide interpretable explanations of model predictions.</p><b>Classes</b><p>BRCA · COAD · KIRC · LUAD · PRAD</p></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><h3>Technology stack</h3><p style="color:#718096">Python · Streamlit · Pandas · scikit-learn · SHAP · Plotly</p><b>Model</b><p>Linear Support Vector Machine (project pipeline)</p></div>',unsafe_allow_html=True)
    st.warning("Educational and research use only. Not validated for clinical decision-making.")

